class APIError(Exception):
    status_code = 500
    message = "Internal server error"

    def __init__(self, message: str | None = None):
        if message:
            self.message = message


class NotFoundError(APIError):
    status_code = 404
    message = "Resource not found"


class TimeoutError(APIError):
    status_code = 504
    message = "Request timed out"


class ValidationError(APIError):
    status_code = 400
    message = "Invalid request"


class DatabaseError(APIError):
    status_code = 500
    message = "Database error"
