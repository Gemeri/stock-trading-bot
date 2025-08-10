import os

# paper trading / SF account
API_KEY   = os.getenv("ALPACA_API_KEY", "PKX2YGROYCB3773UTOFP")
API_SECRET= os.getenv("ALPACA_API_SECRET", "Fd2ARJkeU36FPyugrzKKVmK4SUMRHAQK4ifql9UX")
API_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
