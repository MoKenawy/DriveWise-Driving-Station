import numpy as np

class RatingModule:
    MAX_VIOLATIONS = 600 # in-place of total time driven
    def __init__(self, firebase_interface):
        self.firebase_interface = firebase_interface

    def calculate_rating(self, violations):
        # Assuming min(violations) = 0 and max(violations) = 1000
        normalized_violations = min(violations, self.MAX_VIOLATIONS)
        rating = 5 - (normalized_violations / 10)
        return max(0, rating)  # Ensure rating is between 0 and 5

    def update_driver_rating(self, driver_id):
        # Fetch number of violations for the driver from the database
        violations_count = self.get_violations_count(driver_id)
        
        # Calculate the rating
        rating = self.calculate_rating(violations_count)
        rating = np.round(rating, 2)

        # Update the driver's rating in the database
        self.firebase_interface.update_driver_rating(driver_id, rating)

    def get_violations_count(self, driver_id):
        # Fetch the violations count for the driver from the database
        violations_count = 0
        driver_data = self.firebase_interface.get_driver_data(driver_id)
        if 'violations' in driver_data:
            violations_count = len(driver_data['violations'])
        return violations_count
