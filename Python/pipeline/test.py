a, b, c, d = 1, 0, 3, 4

# 1. Are all zero?
all_zero = all(x == 0 for x in (a, b, c, d))
# equivalent to: a == 0 and b == 0 and c == 0 and d == 0

# 2. Are none zero?
none_zero = all(x != 0 for x in (a, b, c, d))
# equivalent to: a != 0 and b != 0 and c != 0 and d != 0

# 3. Is any zero?
any_zero = any(x == 0 for x in (a, b, c, d))
# equivalent to: a == 0 or b == 0 or c == 0 or d == 0

print(all_zero, none_zero, any_zero)
