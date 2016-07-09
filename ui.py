import model
import stylometry
import accuracy
import os

UIS_LOCATION = './ui/'
ALGORYTHMS = {1: "Multinomial Nayve Bayes Classifier", 2: "Logistic Regression Classifier", 3: "K-Nearest-Neighbors Classifier",
4: "Linear SVC Classifier", 5: "Vote Classifier"}
# This will be PyQT4 one day yay.

class Ui:
    def __init__(self, models, text_analysis, accuracy):
        self._models = models
        self._text_analysis = text_analysis
        self._accuracy = accuracy
        self.initialize_menus()
        self.initialize_menu_choices()
        self.main_menu()

    def user_choice(self, options):
        choice = input('Select an option: ')
        while not choice.isnumeric() or int(choice) not in range(1, options):
            choice = input('No such option! Try again: ')
        return int(choice)
    
    def main_menu(self):
        print(self._main_menu)
        choice = self.user_choice(5)
        self._main_menu_choices[choice]()

    def prediction_menu(self): 
        print(self._prediction_menu)
        choice = self.user_choice(4)
        self._prediction_menu_choices[choice]()

    def expand_library_menu(self):
        print(self._library_menu)
        choice = self.user_choice(5)
        self._library_menu_choices[choice]()

    def exit(self):
        print("Goodbye!")

    def add_author(self):
        author_name = input('Enter author name: ')
        text_files_str = input('Enter authors text(s): ')
        text_files = text_files_str.split(' ')
        result = self._text_analysis.add_author(author_name, set(text_files))  
        self.main_menu()
       
    def predict(self):
        print(self._algorythm_menu)
        choice = self.user_choice(6)
        algorythm = ALGORYTHMS[choice]
        book = str(input('Enter book: '))

        result = self._text_analysis.text_features(book)
        if result is None:
            print("Something went wrong! Please try again!")
            self.prediction_menu()
        else:
            print("Author is: ", self._models.predict((result, algorythm)))
            self.prediction_menu()

    def display_accuracy(self):
        self._accuracy.test_accuracy()
        self.main_menu()

    def display_authors(self):
        print(sorted(self._text_analysis.get_authors()))
        self.main_menu()

    def initialize_menus(self):
        self._main_menu = self.load_menu('main_menu.txt')
        self._library_menu = self.load_menu('library_menu.txt')
        self._prediction_menu = self.load_menu('prediction_menu.txt')
        self._algorythm_menu = self.load_menu('algorythm_menu.txt')

    def initialize_menu_choices(self):
        self._main_menu_choices = {1: self.prediction_menu, 2: self.expand_library_menu, 3: self.display_accuracy, 4: self.exit}
        self._library_menu_choices = {1: self.add_author, 2: self.display_authors, 3: self.main_menu, 4: self.exit}
        self._prediction_menu_choices = {1: self.predict, 2: self.main_menu, 3: self.exit}

    def load_menu(self, menu_name):
        menu_location = os.path.join(UIS_LOCATION, menu_name)
        try:
            with open(menu_location, 'r') as menu_file:
                menu = menu_file.read()
            return menu
        except IOError:
            print("Could not load menus!")
        