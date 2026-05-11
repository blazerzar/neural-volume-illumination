"""
Launches a WebSocket server that trains a model on incoming data. The server
responds with validation metrics or the trained model itself upon request.

Usage:
    python server.py <parameters_path> [options]

Options:
    --show-images                    Display images
"""

import asyncio
import io
import json
import logging
import multiprocessing as mp
import queue as queue_mod
import signal
import time
from sys import argv, exit

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from radiance_field_network import RadianceFieldNetwork
from utils import PIXEL_VALUES, create_image, export_state_dict
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

WS_PORT = 8001

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d [%(levelname)s]: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logging.getLogger('websockets').setLevel(logging.WARNING)
logger = logging.getLogger('server')

client = None

model_parameters = {}
show_images = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


async def main():
    async with serve(handler, '', WS_PORT, max_size=None) as server:
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGTERM, server.close)
        logger.info(f'Listening on port {WS_PORT}')
        await server.wait_closed()


async def handler(websocket):
    global client

    if client is not None:
        logger.warning('Connection rejected: client already connected')
        await websocket.close()
        return

    client = websocket
    logger.info('Client connected')

    model_args = model_parameters['model_args']
    lr = model_parameters['lr']
    model, optimizer = create_model(model_args, lr, device)
    response = json.dumps({'type': 'model-created', 'model_args': model_args})
    await websocket.send(response.encode('utf-8') + b'\0')
    logger.info('Model created')

    train_lock = asyncio.Lock()
    frame = 0

    while True:
        try:
            message = await websocket.recv()
            null_separator = message.index(0)
            msg_type = message[:null_separator].decode('utf-8')

            if msg_type == 'ping':
                data = json.loads(message[null_separator + 1 :].decode('utf-8'))
                logger.info('Ping received')
                response = json.dumps({'type': 'pong', 'time': data['time']})
                await websocket.send(response.encode('utf-8') + b'\0')
            elif msg_type == 'ground-truth':
                if train_lock.locked():
                    logger.warning('Discarding ground truth: training in progress')
                    continue

                asyncio.create_task(
                    train_and_respond(
                        train_lock,
                        model,
                        optimizer,
                        message[null_separator + 1 :],
                        websocket,
                        frame,
                    )
                )
                frame += 1
            elif msg_type == 'model-reset':
                frame = 0
                model, optimizer = create_model(model_args, lr, device)
                logger.info('Model state reset')
                response = json.dumps({'type': 'model-reset'})
                await websocket.send(response.encode('utf-8') + b'\0')
            elif msg_type == 'model-request':
                buffer = io.BytesIO()
                export_state_dict(model, model_args, buffer=buffer)
                model_bytes = buffer.getvalue()
                size_mb = len(model_bytes) / 1024 / 1024
                response = json.dumps({'type': 'model-weights'})
                await websocket.send(response.encode('utf-8') + b'\0' + model_bytes)
                logger.info(f'Sending model ({size_mb:.2f} MB) to client')
        except (ConnectionClosedOK, ConnectionClosedError):
            logger.info('Client disconnected')
            break
        finally:
            client = None


async def train_and_respond(lock, model, optimizer, raw_data, websocket, frame):
    async with lock:
        *data, parsing = await asyncio.to_thread(parse_ground_truth, raw_data)
        if not len(data[0]):
            response = json.dumps({'type': 'metrics', 'val_loss': None})
            await websocket.send(response.encode('utf-8') + b'\0')
            return
        val_loss, training = await asyncio.to_thread(
            train_model, model, optimizer, *data
        )
        try:
            response = json.dumps({'type': 'metrics', 'val_loss': val_loss})
            await websocket.send(response.encode('utf-8') + b'\0')
            logger.info(
                f'Frame: {frame}, '
                f'parsing: {parsing:.2f} ns, '
                f'training: {training:.2f} ms, '
                f'val_loss: {val_loss:.5f}'
            )
        except (ConnectionClosedOK, ConnectionClosedError):
            logger.info('Client disconnected during training')


def create_model(model_args, lr, device):
    model = RadianceFieldNetwork(**model_args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer


def parse_ground_truth(data_bytes):
    start = time.perf_counter()

    # Convert to bytearray to make it writable for PyTorch
    array = torch.frombuffer(bytearray(data_bytes), dtype=torch.float32).to(device)
    array = array.reshape(-1, PIXEL_VALUES)

    # Preprocess
    non_zero_mask = (array != 0).any(dim=1)
    non_nan_mask = ~torch.isnan(array).any(dim=1)
    in_bounds_mask = ((array[:, :3] > 0) & (array[:, :3] < 1)).all(dim=1)
    valid_mask = non_zero_mask & non_nan_mask & in_bounds_mask
    array = array[valid_mask]

    # Normalize azimuth and elevation to [0, 1]
    array[:, 3] = (array[:, 3] + torch.pi) / (2.01 * torch.pi)
    array[:, 4] = array[:, 4] / torch.pi

    X = array[:, :5]
    y = array[:, 5:]
    duration = time.perf_counter() - start

    return X, y, valid_mask.detach().cpu().numpy(), duration


def train_model(model, optimizer, X, y, mask):
    loss_fn = nn.MSELoss(reduction='mean')

    # Validation
    model.eval()
    predicted = model(X)
    val_loss = loss_fn(predicted, y).item()

    # Training
    start = time.perf_counter()
    model.train()
    optimizer.zero_grad()
    outputs = model(X)
    loss = loss_fn(outputs, y)
    loss.backward()
    optimizer.step()
    duration = (time.perf_counter() - start) * 1000

    if show_images:
        resolution = int(np.sqrt(len(mask)))
        original = create_image(y, mask, resolution)
        predicted_image = create_image(predicted, mask, resolution)
        combined = np.hstack([original, predicted_image]).astype(np.float32)
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        frame_queue.put(combined)

    return val_loss, duration


def display_process(frame_queue):
    cv2.namedWindow('Radiance Field Network', cv2.WINDOW_NORMAL)
    while True:
        try:
            frame = frame_queue.get(timeout=0.03)
            if frame is None:
                break
            cv2.imshow('Radiance Field Network', frame)
        except queue_mod.Empty:
            pass
        except Exception as e:
            logger.exception(e)
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if len(argv) < 2:
        print(__doc__)
        exit(1)

    with open(argv[1], 'r') as f:
        model_parameters = json.load(f)

    for i in range(2, len(argv)):
        show_images |= argv[i] == '--show-images'

    display_proc = None
    if show_images:
        frame_queue = mp.Queue()
        display_proc = mp.Process(
            target=display_process, args=(frame_queue,), daemon=True
        )
        display_proc.start()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        if display_proc is not None:
            frame_queue.put(None)
            display_proc.join(timeout=2)
