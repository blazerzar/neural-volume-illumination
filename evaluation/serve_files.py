import http.server
import os


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if not self.path.startswith('/data'):
            self.send_error(404)
            return
        super().do_GET()

    def do_HEAD(self):
        if not self.path.startswith('/data'):
            self.send_error(404)
            return
        super().do_HEAD()

    def translate_path(self, path):
        # Strip the /data prefix so /data/radiance/foo.zip -> radiance/foo.zip
        if path.startswith('/data'):
            path = path[len('/data'):]
        return super().translate_path(path)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()


if __name__ == '__main__':
    directory = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.chdir(directory)
    http.server.test(HandlerClass=CORSHandler, port=8000)
