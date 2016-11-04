from collections import namedtuple


Instance = namedtuple("Instance", "sky air_temp humidity wind water forecast enjoy_sport")


data = (Instance("Sunny", "Warm", "Normal", "Strong", "Warm", "Same", "Yes"),
        Instance("Sunny", "Warm", "High", "Strong", "Warm", "Same", "Yes"),
        Instance("Rainy", "Cold", "High", "Strong", "Warm", "Change", "No"),
        Instance("Sunny", "Warm", "High", "Strong", "Cool", "Change", "Yes"))


len_attributes = len(data[0])
