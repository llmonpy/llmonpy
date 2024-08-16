if __name__ == "__main__":
    profile = []
    factor = 0.0
    factor_increment = 0.0333
    for index in range(60):
        factor += factor_increment
        if factor > 1.0:
            factor = 1.0
        profile.append(factor)
    print("done")
    print(profile)