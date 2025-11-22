import math
from decimal import Decimal, getcontext
import math

class DecimalCustomFloat:
    def __init__(self, value, mantissa_digits=4):
        """
        Initialize the DecimalCustomFloat with specified mantissa digit length.

        Parameters:
        - value (float or Decimal): The number to represent.
        - mantissa_digits (int): Number of digits in the mantissa.
        """
        self.mantissa_digits = mantissa_digits
        self.value = Decimal(str(value))
        self.sign = '-' if self.value < 0 else '+'
        self.exponent = 0
        self.mantissa = Decimal('0')

        if self.value.is_zero():
            self.exponent = 0
            self.mantissa = Decimal('0')
        else:
            self._normalize()

    def _normalize(self):
        # Set precision for Decimal operations
        getcontext().prec = self.mantissa_digits + 5  # Extra precision

        abs_value = self.value.copy_abs()

        # Determine exponent b
        self.exponent = abs_value.adjusted() + 1

        # Compute unrounded mantissa
        mantissa = abs_value.scaleb(-self.exponent)

        # Get mantissa digits up to m+1 places
        mantissa_str = format(mantissa, 'f')
        if '.' in mantissa_str:
            _, fractional_part = mantissa_str.split('.')
        else:
            fractional_part = ''

        # Ensure enough digits
        fractional_part = fractional_part.ljust(self.mantissa_digits + 1, '0')
        mantissa_digits_str = fractional_part[:self.mantissa_digits + 1]

        # Extract digits
        mantissa_m = mantissa_digits_str[:self.mantissa_digits]
        digit_m_plus_1 = int(mantissa_digits_str[self.mantissa_digits])

        # Create mantissa
        mantissa_rounded = Decimal('0.' + mantissa_m)

        # Apply rounding
        if digit_m_plus_1 >= 5:
            increment = Decimal('1e-' + str(self.mantissa_digits))
            mantissa_rounded += increment

        # Handle mantissa overflow
        if mantissa_rounded >= Decimal('1'):
            mantissa_rounded = Decimal('0.1')
            self.exponent += 1

        self.mantissa = mantissa_rounded.normalize()

    def __repr__(self):
        return f"{self.sign}{self.mantissa} * 10^{self.exponent}"

    def to_decimal(self):
        """
        Convert the DecimalCustomFloat back to a Decimal number.
        """
        sign = -1 if self.sign == '-' else 1
        return sign * self.mantissa * (Decimal('10') ** self.exponent)

    # Arithmetic Operations with Rounding
    def _operate(self, other, operation):
        if isinstance(other, DecimalCustomFloat):
            value_other = other.to_decimal()
            mantissa_digits = min(self.mantissa_digits, other.mantissa_digits)
        else:
            value_other = Decimal(str(other))
            mantissa_digits = self.mantissa_digits

        # Perform the operation
        getcontext().prec = mantissa_digits + 5  # Extra precision
        result_value = operation(self.to_decimal(), value_other)

        # Round the result using the same mantissa_digits
        return DecimalCustomFloat(result_value, mantissa_digits)

    def __add__(self, other):
        return self._operate(other, lambda x, y: x + y)

    def __sub__(self, other):
        return self._operate(other, lambda x, y: x - y)

    def __mul__(self, other):
        return self._operate(other, lambda x, y: x * y)

    def __truediv__(self, other):
        return self._operate(other, lambda x, y: x / y)

    # Reverse operations to support operations where DecimalCustomFloat is on the right
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return DecimalCustomFloat(other, self.mantissa_digits).__sub__(self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return DecimalCustomFloat(other, self.mantissa_digits).__truediv__(self)

    # Comparison Operations
    def __lt__(self, other):
        if isinstance(other, DecimalCustomFloat):
            return self.to_decimal() < other.to_decimal()
        else:
            return self.to_decimal() < Decimal(str(other))

    def __gt__(self, other):
        if isinstance(other, DecimalCustomFloat):
            return self.to_decimal() > other.to_decimal()
        else:
            return self.to_decimal() > Decimal(str(other))

    def __eq__(self, other):
        if isinstance(other, DecimalCustomFloat):
            return self.to_decimal() == other.to_decimal()
        else:
            return self.to_decimal() == Decimal(str(other))

    def __le__(self, other):
        return self < other or self == other

    def __ge__(self, other):
        return self > other or self == other

    def __ne__(self, other):
        return not self == other


class CustomFloat:
    def __init__(self, value, mantissa_bits=23):
        """
        Initialize the CustomFloat with a specified mantissa length.

        Parameters:
        - value (float): The floating-point number to represent.
        - mantissa_bits (int): Number of bits in the mantissa.
        """
        self.mantissa_bits = mantissa_bits
        self.value = float(value)
        self.sign, self.exponent, self.mantissa = self.float_to_components(self.value)
    
    def float_to_components(self, value):
        """
        Decompose the float into sign, exponent, and mantissa.

        Returns:
        - sign (int): 0 for positive, 1 for negative.
        - exponent (int): Exponent value.
        - mantissa (float): Mantissa value.
        """
        if value == 0.0:
            return 0, 0, 0.0
        
        sign = 0
        if value < 0:
            sign = 1
            value = -value
        
        exponent = math.floor(math.log2(value))
        mantissa = value / (2 ** exponent) - 1  # Normalize mantissa to [0, 1)
        
        return sign, exponent, mantissa
    
    def components_to_float(self, sign, exponent, mantissa):
        """
        Reconstruct the float from sign, exponent, and mantissa.

        Returns:
        - value (float): The reconstructed floating-point number.
        """
        return ((-1) ** sign) * (1 + mantissa) * (2 ** exponent)
    
    def quantize_mantissa(self):
        """
        Quantize the mantissa to the specified number of bits.

        Returns:
        - quantized_mantissa (float): The mantissa after quantization.
        """
        quantization_step = 1 / (2 ** self.mantissa_bits)
        quantized_mantissa = math.floor(self.mantissa / quantization_step) * quantization_step
        return quantized_mantissa
    
    def to_custom_float(self):
        """
        Convert the internal components to the quantized custom float.

        Returns:
        - custom_value (float): The custom floating-point number with specified mantissa bits.
        """
        quantized_mantissa = self.quantize_mantissa()
        return self.components_to_float(self.sign, self.exponent, quantized_mantissa)
    
    def __repr__(self):
        custom_value = self.to_custom_float()
        return f"CustomFloat(value={custom_value}, mantissa_bits={self.mantissa_bits})"
    
    # Arithmetic Operations
    def __add__(self, other):
        if isinstance(other, CustomFloat):
            result = self.to_custom_float() + other.to_custom_float()
        else:
            result = self.to_custom_float() + other
        return CustomFloat(result, self.mantissa_bits)
    
    def __sub__(self, other):
        if isinstance(other, CustomFloat):
            result = self.to_custom_float() - other.to_custom_float()
        else:
            result = self.to_custom_float() - other
        return CustomFloat(result, self.mantissa_bits)
    
    def __mul__(self, other):
        if isinstance(other, CustomFloat):
            result = self.to_custom_float() * other.to_custom_float()
        else:
            result = self.to_custom_float() * other
        return CustomFloat(result, self.mantissa_bits)
    
    def __truediv__(self, other):
        if isinstance(other, CustomFloat):
            result = self.to_custom_float() / other.to_custom_float()
        else:
            result = self.to_custom_float() / other
        return CustomFloat(result, self.mantissa_bits)
    
    # Comparison Operations
    def __gt__(self, other):
        """
        Greater-than comparison.

        Returns:
        - bool: True if self > other, False otherwise.
        """
        if isinstance(other, CustomFloat):
            return self.to_custom_float() > other.to_custom_float()
        else:
            return self.to_custom_float() > other
    
    def __lt__(self, other):
        """
        Less-than comparison.

        Returns:
        - bool: True if self < other, False otherwise.
        """
        if isinstance(other, CustomFloat):
            return self.to_custom_float() < other.to_custom_float()
        else:
            return self.to_custom_float() < other
    
    def __eq__(self, other):
        """
        Equality comparison.

        Returns:
        - bool: True if self == other, False otherwise.
        """
        if isinstance(other, CustomFloat):
            return self.to_custom_float() == other.to_custom_float()
        else:
            return self.to_custom_float() == other
    
    def __ge__(self, other):
        """
        Greater-than or equal comparison.

        Returns:
        - bool: True if self >= other, False otherwise.
        """
        return self > other or self == other
    
    def __le__(self, other):
        """
        Less-than or equal comparison.

        Returns:
        - bool: True if self <= other, False otherwise.
        """
        return self < other or self == other
    
    def __ne__(self, other):
        """
        Not equal comparison.

        Returns:
        - bool: True if self != other, False otherwise.
        """
        return not self == other