#-*- coding:utf8 -*-
import pygame, sys, time
import numpy as np
from pygame.locals import *

#矩阵宽与矩阵高
WIDTH = 80
HEIGHT = 40

#记录鼠标按键情况的全局变量
pygame.button_down = False

#记录游戏世界的矩阵
pygame.world=np.zeros((HEIGHT,WIDTH))

#创建 Cell 类方便细胞绘制
class Cell(pygame.sprite.Sprite):

    size = 10

    def __init__(self, position):

        pygame.sprite.Sprite.__init__(self)

        self.image = pygame.Surface([self.size, self.size])

        #填上白色
        self.image.fill((255,255,255))

        # 创建一个以左上角为锚点的矩形
        self.rect = self.image.get_rect()
        self.rect.topleft = position

#绘图函数，注意到我们是把画布重置了再遍历整个世界地图，因此有很大的性能提升空间
def draw():
    screen.fill((0,0,0))
    for sp_col in range(pygame.world.shape[1]):
        for sp_row in range(pygame.world.shape[0]):
            if pygame.world[sp_row][sp_col]:
                new_cell = Cell((sp_col * Cell.size,sp_row * Cell.size))
                screen.blit(new_cell.image,new_cell.rect)

#根据细胞更新规则更新地图
def next_generation():
    nbrs_count = sum(np.roll(np.roll(pygame.world, i, 0), j, 1)
                 for i in (-1, 0, 1) for j in (-1, 0, 1)
                 if (i != 0 or j != 0))

    pygame.world = (nbrs_count == 3) | ((pygame.world == 1) & (nbrs_count == 2)).astype('int')

#地图初始化
def init():
    pygame.world.fill(0)
    draw()
    return 'Stop'

#停止时的操作
def stop():
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == KEYDOWN and event.key == K_RETURN:
            return 'Move'

        if event.type == KEYDOWN and event.key == K_r:
            return 'Reset'

        if event.type == MOUSEBUTTONDOWN:
            pygame.button_down = True
            pygame.button_type = event.button

        if event.type == MOUSEBUTTONUP:
            pygame.button_down = False

        if pygame.button_down:
            mouse_x, mouse_y = pygame.mouse.get_pos()

            sp_col = mouse_x / Cell.size;
            sp_row = mouse_y / Cell.size;

            if pygame.button_type == 1: #鼠标左键
                pygame.world[sp_row][sp_col] = 1
            elif pygame.button_type == 3: #鼠标右键
                pygame.world[sp_row][sp_col] = 0
            draw()

    return 'Stop'

#计时器，控制帧率
pygame.clock_start = 0


#进行演化时的操作
def move():
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN and event.key == K_SPACE:
            return 'Stop'
        if event.type == KEYDOWN and event.key == K_r:
            return 'Reset'
        if event.type == MOUSEBUTTONDOWN:
            pygame.button_down = True
            pygame.button_type = event.button

        if event.type == MOUSEBUTTONUP:
            pygame.button_down = False

        if pygame.button_down:
            mouse_x, mouse_y = pygame.mouse.get_pos()

            sp_col = mouse_x / Cell.size;
            sp_row = mouse_y / Cell.size;

            if pygame.button_type == 1:
                pygame.world[sp_row][sp_col] = 1
            elif pygame.button_type == 3:
                pygame.world[sp_row][sp_col] = 0
            draw()


    if time.clock() - pygame.clock_start > 0.02:
        next_generation()
        draw()
        pygame.clock_start = time.clock()

    return 'Move'



if __name__ == '__main__':

    #状态机对应三种状态，初始化，停止，进行
    state_actions = {
            'Reset': init,
            'Stop': stop,
            'Move': move
        }
    state = 'Reset'

    pygame.init()
    pygame.display.set_caption('Conway\'s Game of Life')

    screen = pygame.display.set_mode((WIDTH * Cell.size, HEIGHT * Cell.size))

    while True: # 游戏主循环

        state = state_actions[state]()
        pygame.display.update()


