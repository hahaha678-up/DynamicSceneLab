import json
import socket


class HuNavClient:
    def __init__(self, endpoint):
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.settimeout(30.)
        self.socket.connect(str(endpoint))
        self.stream = self.socket.makefile('rwb')

    def request(self, **request):
        self.stream.write((json.dumps(request, allow_nan=False)+'\n').encode())
        self.stream.flush()
        line = self.stream.readline(65537)
        if not line or len(line) > 65536:
            raise ConnectionError('HuNav bridge disconnected or oversized reply')
        response = json.loads(line)
        if not response.get('ok'):
            raise RuntimeError(response.get('error', 'HuNav bridge failure'))
        return response

    def close(self):
        self.stream.close()
        self.socket.close()
