import sys,random

n = int(sys.argv[1])
for i in range(n):
    print(random.randrange(0, 100))

def main():
    print(sys.argv[1])
    print(sys.argv[2])

if __name__ == '__main__':
    main()