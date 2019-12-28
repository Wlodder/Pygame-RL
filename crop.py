import pygame
import random

class wheat(pygame.Rect):

    score = 10
    pulled = False
    seen = False
    color = (0,0,0)

def plant_crop(crops):
    pos_x = random.randint(50,650)
    pos_y = random.randint(50,450)
    crops.append(wheat(pos_x,pos_y,50,50))