import json
from pathlib import Path
from typing import Optional, Union

currency_file_path = Path(__file__).parent / "../../config/currency_conversion.json"
with currency_file_path.open("r") as currency_file:
    currency_conversion = json.load(currency_file)


def get_supported_currency():
    return currency_conversion.keys()


def convert_currency(amount: Optional[Union[int, float]], src: str, dst: str) -> Optional[Union[int, float]]:
    if src != dst and amount is not None:
        conversion_base = currency_conversion[src]['base']
        conversion_rate = currency_conversion[src]['conversion'][dst]
        return amount / conversion_base * conversion_rate
    else:
        return amount
