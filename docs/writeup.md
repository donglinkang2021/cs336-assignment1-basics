## Problem (unicode1): Understanding Unicode (1 point)

**(a) What Unicode character does `chr(0)` return?**
*Deliverable: A one-sentence response.*

**My:** `chr(0)` returns the null byte `'\x00'`.

**(b) How does this character’s string representation (`__repr__()`) differ from its printed representation?**
*Deliverable: A one-sentence response.*

**My:** `chr(0).__repr__()` returns `'\\x00'`, adding a backslash `\` before the `x`.

**(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:**
```python
>>> chr(0)
>>> print(chr(0))
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```
*Deliverable: A one-sentence response.*

**My:** It prints nothing when it occurs in text; the result is "this is a teststring".

## Problem (unicode2): Unicode Encodings (3 points)

**(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.**
*Deliverable: A one-to-two sentence response.*

**My:** UTF-8 is more space-efficient for texts that are primarily in ASCII, as it uses one byte for these characters, while UTF-16 and UTF-32 use two and four bytes respectively; additionally, UTF-8 is backward compatible with ASCII and is the most widely used encoding on the web(As mentioned before, UTF-8 is the dominant encoding in the Internet (more than 98% of all webpages)).

**(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.**
```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
```
```python
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```
*Deliverable: An example input byte string for which `decode_utf8_bytes_to_str_wrong` produces incorrect output, with a one-sentence explanation of why the function is incorrect.*

**My:** An example input byte string is `b'\xc3\xa9'`, which represents the character 'é' in UTF-8. The function is incorrect because **it decodes each byte individually**, leading to a `UnicodeDecodeError` since `b'\xc3'` and `b'\xa9'` are not valid standalone UTF-8 characters.

```python
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])

print("hello".encode("utf-8"))
print("hello".encode("utf-8").decode("utf-8"))
print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
print("é".encode("utf-8"))
print("é".encode("utf-8").decode("utf-8"))
print(decode_utf8_bytes_to_str_wrong("é".encode("utf-8")))

# b'hello'
# hello
# hello
# b'\xc3\xa9'
# é
# Traceback (most recent call last):
#   File "assignment1-basics/demo.py", line 9, in <module>
#     print(decode_utf8_bytes_to_str_wrong("é".encode("utf-8")))
#           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
#   File "assignment1-basics/demo.py", line 2, in decode_utf8_bytes_to_str_wrong
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])
#                     ~~~~~~~~~~~~~~~~~^^^^^^^^^
# UnicodeDecodeError: 'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data
```

**(c) Give a two byte sequence that does not decode to any Unicode character(s).**
*Deliverable: An example, with a one-sentence explanation.*

**My:** The byte sequence `b'\x80\x80'` does not decode to any Unicode characters because it is an invalid UTF-8 sequence; in UTF-8, continuation bytes (bytes starting with `10xxxxxx`) must follow a valid leading byte, and `0x80` cannot be a leading byte.