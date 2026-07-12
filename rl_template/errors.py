class EmptyBufferError(Exception):

    def  __init__(self, current_size:int, require_buffer_size:int):
        self.current_size = current_size 
        self.require_buffer_size = require_buffer_size

        self.message = "Training agent flow is incorrect: the buffer is empty"
        super().__init__(self.message)


        def __str__(self):
            suggestion = "Call Rollout before update weights"
            details = f"[Crash Workflow] {self.message}\n"
            details += f"the current buffer size: {self.current_size}\n"
            details += f" the minimal size required is: {self.require_buffer_size}\n"
            details += f"{suggestion}"
            return details
