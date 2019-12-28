import pygame

class farmer(pygame.Rect):

    moved = False
    color = (255,0,0)
    life = 0

    def defineVision(self,width,height):
        self.visionRect = pygame.Rect(self.x,self.y,width,height)

    def move(self, x, y):
        self.x += x
        self.y += y
        self.visionRect.x += x
        self.visionRect.y += y
        self.moved = True
        

    def move_right(self,value):
        self.move(value, 0)

    def move_left(self,value):
        self.move(-value, 0)

    def move_up(self,value):
        self.move(0, -value)

    def move_down(self,value):
        self.move(0, value)

    def process(self,action):
        move_value = 100
        if action == 0:
            self.move_right(move_value)
        elif action == 1:
            self.move_left(move_value)
        elif action == 2:
            self.move_down(move_value)
        else:
            self.move_up(move_value)
        self.life += 0.1

    