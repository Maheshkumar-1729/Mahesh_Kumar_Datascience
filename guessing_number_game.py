import random  # random module has random generation related functions
num = random.randint(1,100)
attempts = 0
is_invalid = False
while True:  # infinite loop

    guess = int(input("Guess the number (1-100): "))
    if guess <1 or guess>100:
        if is_invalid:
            print("You lose! This is your second invalid attempt!")
            break
        else:
            is_invalid = True
            print("This is your first invalid attempt, one more will end the game!")
            continue
    attempts+=1
    if guess == num:
        print(f"Congratulations! You guessed it correctly in {attempts} attempts!")
        break
    elif guess< num:
        print("Sorry! thats incorrect, Try again with a higher number... ")
    else:
        print("Sorry! thats incorrect, Try again with a lower number... ")
