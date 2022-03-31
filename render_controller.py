import pygame, sys
import os, sys
import constants
import pygame

def get_tile_color(tile_contents):
    tile_color = constants.GOLD
    if tile_contents == "m":
        tile_color = constants.DARKGREY
    if tile_contents == 0:
        tile_color = constants.GREEN
    if tile_contents == 2:
        tile_color = constants.BLACK
    if tile_contents == 1:
        tile_color = constants.RED
    return tile_color

def draw_map(surface, map_tiles):
    for j, tile in enumerate(map_tiles):
        for i, tile_contents in enumerate(tile):
            # print("{},{}: {}".format(i, j, tile_contents))
            myrect = pygame.Rect(i*constants.BLOCK_WIDTH, j*constants.BLOCK_HEIGHT, constants.BLOCK_WIDTH, constants.BLOCK_HEIGHT)
            pygame.draw.rect(surface, get_tile_color(tile_contents), myrect)

def draw_grid(surface):
    for i in range(constants.NUMBER_OF_BLOCKS_WIDE):
        new_height = round(i * constants.BLOCK_HEIGHT)
        new_width = round(i * constants.BLOCK_WIDTH)
        pygame.draw.line(surface, constants.BLACK, (0, new_height), (constants.SCREEN_WIDTH, new_height), 2)
        pygame.draw.line(surface, constants.BLACK, (new_width, 0), (new_width, constants.SCREEN_HEIGHT), 2)

def game_loop(surface, world_map):
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
        draw_map(surface, world_map)
        draw_grid(surface)
        pygame.display.update()

def initialize_game():
    pygame.init()
    surface = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
    pygame.display.set_caption(constants.TITLE)
    surface.fill(constants.UGLY_PINK)
    return surface

#def read_map(observation):

    # filepath = os.path.join("data", constants.MAPFILE)
    # with open(filepath, 'r') as f:
    #     world_map = f.readlines()
    # world_map = [line.strip() for line in world_map]
    #return (world_map)

# def main():
#     #world_map = read_map(observation)
#     surface = initialize_game()
#     game_loop(surface, world_map)

if __name__=="__main__":
    #main()
    pass