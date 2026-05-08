import socket
import unittest
from unittest import mock

from r_peak_detection import visualize


class VisualizePortTests(unittest.TestCase):
    def test_is_port_available_detects_busy_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((visualize.DEFAULT_HOST, 0))
            sock.listen()
            busy_port = sock.getsockname()[1]

            self.assertFalse(visualize.is_port_available(visualize.DEFAULT_HOST, busy_port))

    def test_resolve_port_raises_when_explicit_port_is_busy(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((visualize.DEFAULT_HOST, 0))
            sock.listen()
            busy_port = sock.getsockname()[1]

            with self.assertRaises(OSError):
                visualize.resolve_port(visualize.DEFAULT_HOST, busy_port)

    def test_resolve_port_rejects_invalid_port(self):
        with self.assertRaises(ValueError):
            visualize.resolve_port(visualize.DEFAULT_HOST, 70000)

    def test_stop_dash_server_in_current_session_stops_registered_server(self):
        server = mock.Mock()
        server_key = (visualize.DEFAULT_HOST, "8050")
        fake_dash = mock.Mock()
        fake_dash.jupyter_dash._servers = {server_key: server}

        with mock.patch.dict("sys.modules", {"dash": fake_dash}):
            stopped = visualize.stop_dash_server_in_current_session(
                visualize.DEFAULT_HOST, 8050
            )

        self.assertTrue(stopped)
        server.shutdown.assert_called_once_with()
        self.assertNotIn(server_key, fake_dash.jupyter_dash._servers)

    def test_find_available_port_skips_busy_preferred_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((visualize.DEFAULT_HOST, 0))
            sock.listen()
            busy_port = sock.getsockname()[1]

            free_port = visualize.find_available_port(
                visualize.DEFAULT_HOST, busy_port
            )

        self.assertGreater(free_port, busy_port)


if __name__ == "__main__":
    unittest.main()
