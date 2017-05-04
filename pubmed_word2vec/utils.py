def get_journals():
    with open('journals.txt') as f:
        journals=[i.strip() for i in f.readlines()]
    return journals
