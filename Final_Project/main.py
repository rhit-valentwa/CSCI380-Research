# Uses Python 3.10.4
# Modules:
#   cv2
#   mediapipe
#   math
#   tkinter

# --- imports ---

import cv2
import mediapipe as mp
import math
import tkinter

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

root = tkinter.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# initialize video capture object to read video from external webcam
cap = cv2.VideoCapture(1)
# if there is no external camera then take the built-in camera
if not cap.read()[0]:
    cap = cv2.VideoCapture(0)

tileSize = 0 # default value
selectorSize = 20

players = [{'id': 1, 'x': 0, 'y': 0, 'symbol': 'X', 'old_index_distance': 0, 'index_distance': 0, 'is_clicking': False, 'color': (0, 255, 0)},
           {'id': 2, 'x': 0, 'y': 0, 'symbol': 'O', 'old_index_distance': 0, 'index_distance': 0, 'is_clicking': False, 'color': (0, 0, 255)}]

playerWon = False
playerTurn = 1
playerWinMessage = ''
winLocations = [[], []]

grid = [
  ['', '', ''],
  ['', '', ''],
  ['', '', ''],
]

def rectangle(x, y, w, h, color, fill):
  cv2.rectangle(image, (x, y), (x + w, y + h), (color), fill)

# finds winner, checks to see if board is full
def checkWin():
  global playerWon
  global playerWinMessage
  global winLocations
  
  winning_positions = [[(0, 0), (0, 1), (0, 2)],
                       [(1, 0), (1, 1), (1, 2)],
                       [(2, 0), (2, 1), (2, 2)],
                       [(0, 0), (1, 0), (2, 0)],
                       [(0, 1), (1, 1), (2, 1)],
                       [(0, 2), (1, 2), (2, 2)],
                       [(0, 0), (1, 1), (2, 2)],
                       [(2, 0), (1, 1), (0, 2)]]

  # run loop through winning_positions
  for position in winning_positions:
      for playerIndex in range(0, 2):
        #if grid has a winning position
        if grid[position[0][1]][position[0][0]] == players[playerIndex]['symbol']\
          and grid[position[1][1]][position[1][0]] == players[playerIndex]['symbol']\
          and grid[position[2][1]][position[2][0]] == players[playerIndex]['symbol']:
          playerWon = True
          playerWinMessage = 'Player ' + str(playerIndex + 1) + ' won!'
          winLocations = [[position[0][0], position[0][1]],
                          [position[2][0], position[2][1]]]

def drawGrid(image):
  global playerTurn
  for y in range(0, 3):
      for x in range(0, 3):
        color = (100, 100, 100)

        for player in players:
          playerX = player['x']
          playerY = player['y']

          squareX = int(image.shape[1] / 3) + (tileSize * x)
          squareY = int(image.shape[0] / 5) + (tileSize * y)

          if playerX <= squareX + tileSize and\
            playerX + selectorSize >= squareX and\
            playerY <= squareY + tileSize and\
            playerY + selectorSize >= squareY and\
            playerWon == False:
            if player['is_clicking'] and grid[y][x] == '' and player['id'] == playerTurn:
              color = (75, 75, 75)
              grid[y][x] = player['symbol']
              checkWin()
              player['is_clicking'] = False
              if playerTurn == 1:
                playerTurn = 2
              else:
                playerTurn = 1
            break
          
        cv2.rectangle(image, (squareX, squareY), (squareX + tileSize, squareY + tileSize), color, -1)
        cv2.rectangle(image, (squareX, squareY), (squareX + tileSize, squareY + tileSize), (0, 0, 0), 2)

        cv2.putText(image, grid[y][x], (squareX + int(tileSize / 4), squareY + int(tileSize / 1.5)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2, cv2.LINE_AA)

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.9,
    min_tracking_confidence=0.5,
    ) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if tileSize == 0:
      tileSize = int(image.shape[1] / 8)

    cv2.rectangle(image, (0, 0), (screen_width, screen_height), (100, 100, 100), -1)

    index = 0

    drawGrid(image)

    # removes the outer lines of boxes on the edges of the board
    rectangle(int(image.shape[1] / 3) + (tileSize * 3) - 2, int(image.shape[0] / 5) - 1, 3, tileSize * 3 + 2, (100, 100, 100), -1)
    rectangle(int(image.shape[1] / 3) - 2, int(image.shape[0] / 5) - 1, 3, tileSize * 3 + 2, (100, 100, 100), -1)
    rectangle(int(image.shape[1] / 3) - 2, int(image.shape[0] / 5) - 1, tileSize * 3 + 2, 3, (100, 100, 100), -1)
    rectangle(int(image.shape[1] / 3) - 2, int(image.shape[0] / 5) + (tileSize * 3) - 1, tileSize * 3 + 2, 3, (100, 100, 100), -1)

    # check distance between points 9 and 6 (index finger) (if has decreased)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        keypoints = []

        shape_color = players[index]['color']

        if index >= 2:
          break

        for data_point in hand_landmarks.landmark:
          keypoints.append({'x': data_point.x, 'y': data_point.y})

        players[index]['x'] = int(keypoints[10]['x'] * image.shape[1])
        players[index]['y'] = int(keypoints[10]['y'] * image.shape[0])
        
        players[index]['old_index_distance'] = players[index]['index_distance']
        players[index]['index_distance'] = math.sqrt((keypoints[8]['x'] - keypoints[5]['x'])**2 + (keypoints[8]['y'] - keypoints[5]['y'])**2)

        if players[index]['index_distance'] + 0.035 < players[index]['old_index_distance']:
          players[index]['is_clicking'] = True
        
        cv2.rectangle(image, (players[index]['x'], players[index]['y']), (players[index]['x'] + selectorSize, players[index]['y'] + selectorSize), shape_color, -1)
        cv2.rectangle(image, (players[index]['x'], players[index]['y']), (players[index]['x'] + selectorSize, players[index]['y'] + selectorSize), (0, 0, 0), 2)
        
        index += 1
    
    if playerWon == True:
      cv2.line(image, ((winLocations[0][0] * tileSize) + int(image.shape[1] / 3) + int(tileSize / 2), (winLocations[0][1] * tileSize) + int(image.shape[0] / 5) + int(tileSize / 2)),\
                      ((winLocations[1][0] * tileSize) + int(image.shape[1] / 3) + int(tileSize / 2), (winLocations[1][1] * tileSize) + int(image.shape[0] / 5) + int(tileSize / 2)), (0, 0, 255), 3)


    image = cv2.resize(image, (screen_width, screen_height))
    image = cv2.flip(image, 1)

    if playerWon == True:
      cv2.putText(image, playerWinMessage, (int(image.shape[1] / 2.8), int(image.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
    else:
      cv2.putText(image, 'Player ' + str(playerTurn) + ' ("' + players[playerTurn - 1]['symbol'] + '") has a turn', (int(image.shape[1] / 4), int(image.shape[0] / 6)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
    

    cv2.imshow('Tic-Tac-Toe', image)  

    if cv2.waitKey(1) == ord('q'):
        break
    
cv2.destroyAllWindows()