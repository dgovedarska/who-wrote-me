import model
import stylometry
import accuracy
import os 

UIS_LOCATION = './ui/'
algos = {1: "Multinomial Nayve Bayes Classifier", 2: "Logistic Regression Classifier", 3: "K-Nearest-Neighbors Classifier",
4: "Linear SVC Classifier", 5: "Vote Classifier"}
# This will be PyQT4 one day yay.

class ui:
    def __init__(self, models, text_analysis, accuracy):
        self._models = models
        self._text_analysis = text_analysis
        self._accuracy = accuracy
        self.initialize_menus()
        self.initialize_menu_choices()
        self.main_menu()
    
    def main_menu(self):
        print(self._main_menu)
        choice = input('Select an option: ')
        self._main_menu_choices[int(choice)]()

    def prediction_menu(self): 
        print(self._prediction_menu)
        choice = input('Select an option: ')
        self._prediction_menu_choices[int(choice)]()

    def expand_library_menu(self):
        print(self._library_menu)
        choice = input('Select an option: ')
        self._library_menu_choices[int(choice)]()

    def exit(self):
        print("Goodbye!")

    def check_bad_input(self): pass

    def add_author(self):
        # TODO massive error handling
        author_name = input('Enter author name: ')
        text_files_str = input('Enter authors text(s): ')
        text_files = text_files_str.split(' ')
        self._text_analysis.add_author(author_name, set(text_files))

        print('Author successfuly added!')
        self.main_menu()

    def predict(self):
        print(self._algorythm_menu)
        choice = int(input('Select an option: '))
        algorythm = algos[choice]
        book = str(input('Enter book: '))

        print("Author is: ", self._models.predict(self._text_analysis.text_features(book), algorythm))
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
        #TODO Error handling
        menu_location = os.path.join(UIS_LOCATION, menu_name)
        menu_file = open(menu_location, 'r')
        menu = menu_file.read()
        menu_file.close()
        return menu
        







