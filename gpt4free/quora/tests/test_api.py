import unittest
import requests
from unittest.mock import MagicMock
from gpt4free.quora.api import retry_request


class TestRetryRequest(unittest.TestCase):
    def test_successful_request(self):
        # Mock a successful request with a 200 status code
        mock_response = MagicMock()
        mock_response.status_code = 200
        requests.get = MagicMock(return_value=mock_response)

        # Call the function and assert that it returns the response
        response = retry_request(requests.get, "http://example.com", max_attempts=3)
        self.assertEqual(response.status_code, 200)

    def test_exponential_backoff(self):
        # Mock a failed request that succeeds after two retries
        mock_response = MagicMock()
        mock_response.status_code = 200
        requests.get = MagicMock(side_effect=[requests.exceptions.RequestException] * 2 + [mock_response])

        # Call the function and assert that it retries with exponential backoff
        with self.assertLogs() as logs:
            response = retry_request(requests.get, "http://example.com", max_attempts=3, delay=1)
            self.assertEqual(response.status_code, 200)
            self.assertGreaterEqual(len(logs.output), 2)
            self.assertIn("Retrying in 1 seconds...", logs.output[0])
            self.assertIn("Retrying in 2 seconds...", logs.output[1])

    def test_too_many_attempts(self):
        # Mock a failed request that never succeeds
        requests.get = MagicMock(side_effect=requests.exceptions.RequestException)

        # Call the function and assert that it raises an exception after the maximum number of attempts
        with self.assertRaises(RuntimeError):
            retry_request(requests.get, "http://example.com", max_attempts=3)
