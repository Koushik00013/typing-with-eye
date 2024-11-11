import cv2
import numpy as np

# Create a keyboard image
keybord = np.zeros((600, 1500), np.uint8)  # Adjusted width to fit more keys

# Define the keys
keys_set_1 = {
    0: "Q", 1: "W", 2: "E", 3: "R", 4: "T",
    5: "A", 6: "S", 7: "D", 8: "F", 9: "G",
    10: "Z", 11: "X", 12: "C", 13: "V", 14: "B",
    15: "Y", 16: "U", 17: "I", 18: "O", 19: "P",
    20: "H", 21: "J", 22: "K", 23: "L",
    24: "N", 25: "M"
}

def letter(letter_index, text, litter_light):
    # Calculate the position of the key based on its index
    x = (letter_index % 10) * 150  # Adjusted x position for more keys
    y = (letter_index // 10) * 200  # Adjusted y position for more keys

    width = 150
    height = 200
    th = 3  # thickness

    # Draw the rectangle based on whether the key is lit or not
    if litter_light:
        cv2.rectangle(keybord, (x + th, y + th), (x + width - th, y + height - th), (255, 255, 255), -1)
    else:
        cv2.rectangle(keybord, (x + th, y + th), (x + width - th, y + height - th), (255, 0, 0), th)

    # Set text properties
    font_letter = cv2.FONT_HERSHEY_PLAIN
    font_scale = 6
    font_th = 4

    # Get the size of the text
    text_size = cv2.getTextSize(text, font_letter, font_scale, font_th)[0]
    width_text, height_text = text_size[0], text_size[1]

    # Calculate text position to center it within the rectangle
    text_x = x + (width - width_text) // 2
    text_y = y + (height + height_text) // 2

    # Put the text on the keyboard
    cv2.putText(keybord, text, (text_x, text_y), font_letter, font_scale, (255, 0, 0), font_th)

# Draw the keys on the keyboard
for i in range(26):  # Now includes all keys from Q to M
    light = (i == 5)  # Light up the key at index 5 (which is "A")#
    letter(i, keys_set_1[i], light)

# Display the keyboard
cv2.imshow("keybord", keybord)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the window