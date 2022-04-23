import sys
import os, sys
from CmaesAgent import CmaesAgent
from HeuristicAgent import HeuristicAgent
from ReinforcedAgent import ReinforcedAgent
import constants
import pygame

def draw_map(surface, map_tiles, num_blocks_w, num_blocks_h):
    BLOCK_WIDTH=constants.SCREEN_WIDTH/num_blocks_w
    BLOCK_HEIGHT=constants.SCREEN_HEIGHT/num_blocks_h
    for j, tile in enumerate(map_tiles):
        for i, tile_contents in enumerate(tile):
            myrect = pygame.Rect(i*BLOCK_WIDTH, j*BLOCK_HEIGHT, BLOCK_WIDTH, BLOCK_HEIGHT)
            pygame.draw.rect(surface, tile_contents, myrect)

def draw_grid(surface, num_blocks_w, num_blocks_h):
    BLOCK_WIDTH=constants.SCREEN_WIDTH/num_blocks_w
    BLOCK_HEIGHT=constants.SCREEN_HEIGHT/num_blocks_h
    for i in range(num_blocks_w):
        new_height = round(i * BLOCK_HEIGHT)
        new_width = round(i * BLOCK_WIDTH)
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

def draw_text(surface, text):
    font = pygame.font.SysFont(None, 12)
    img = font.render(text, True, constants.RED)
    surface.blit(img,(5,5))

def draw_text_multiple_agents(surface, main_env, agents):
    font = pygame.font.SysFont(None, 24)
    agent_color_dict={HeuristicAgent:constants.RED,ReinforcedAgent:constants.GREEN,CmaesAgent:constants.BLUE}
    for e,agent in enumerate(agents):
        # if isinstance(agent, HeuristicAgent):
        #     color=constants.RED
        # elif isinstance(agent, ReinforcedAgent):
        #     color=constants.GREEN
        # else:
        #     color=constants.BLUE
        color=agent_color_dict[type(agent)]
        img = font.render(f"POINTS COLLECTED: {agent.env.points_collected}", True, color)
        #agent.env.pointsCollected
        surface.blit(img,(5,15*e))

def draw_agent(surface, agentPosition, num_blocks_w, num_blocks_h):
    BLOCK_WIDTH=constants.SCREEN_WIDTH/num_blocks_w
    BLOCK_HEIGHT=constants.SCREEN_HEIGHT/num_blocks_h
    myrect = pygame.Rect((agentPosition[1]+1/3)*BLOCK_WIDTH, (agentPosition[0]+1/3)*BLOCK_HEIGHT, (BLOCK_WIDTH/3), (BLOCK_HEIGHT/3))
    pygame.draw.rect(surface, (255,255,255), myrect)

def draw_bases(surface, basesPosition, num_blocks_w, num_blocks_h):
    BLOCK_WIDTH=constants.SCREEN_WIDTH/num_blocks_w
    BLOCK_HEIGHT=constants.SCREEN_HEIGHT/num_blocks_h
    for base in basesPosition:
        myrect = pygame.Rect((base[1]+1/3)*BLOCK_WIDTH, (base[0]+1/3)*BLOCK_HEIGHT, (BLOCK_WIDTH/3), (BLOCK_HEIGHT/3))
        pygame.draw.rect(surface, (0,0,0), myrect)

def draw_bases_multiple_agents(surface, basesPosition, baseColor, num_blocks_w, num_blocks_h):
    BLOCK_WIDTH=constants.SCREEN_WIDTH/num_blocks_w
    BLOCK_HEIGHT=constants.SCREEN_HEIGHT/num_blocks_h
    for e,base in enumerate(basesPosition):
        myrect = pygame.Rect((base[1]+1/3)*BLOCK_WIDTH, (base[0]+1/3)*BLOCK_HEIGHT, (BLOCK_WIDTH/3), (BLOCK_HEIGHT/3))
        myrect_full = pygame.Rect((base[1])*BLOCK_WIDTH, (base[0])*BLOCK_HEIGHT, (BLOCK_WIDTH), (BLOCK_HEIGHT))
        pygame.draw.rect(surface, baseColor[e], myrect_full)
        pygame.draw.rect(surface, (0,0,0), myrect)

def draw_food_multiple_agents(surface, food_list, baseColor, num_blocks_w, num_blocks_h):
    BLOCK_WIDTH=constants.SCREEN_WIDTH/num_blocks_w
    BLOCK_HEIGHT=constants.SCREEN_HEIGHT/num_blocks_h
    for food in food_list:
        myrect = pygame.Rect((food[1])*BLOCK_WIDTH, (food[0])*BLOCK_HEIGHT, (BLOCK_WIDTH), (BLOCK_HEIGHT))
        pygame.draw.rect(surface, baseColor[food[2]], myrect)


def draw_multiple_agents(surface,agents, num_blocks_w,num_blocks_h):
    BLOCK_WIDTH=constants.SCREEN_WIDTH/num_blocks_w
    BLOCK_HEIGHT=constants.SCREEN_HEIGHT/num_blocks_h
    for agent in agents:
        agentPosition=agent.env.agentPosition
        base_colors=agent.env.base_colors
        agent_color_dict={HeuristicAgent:constants.RED,ReinforcedAgent:constants.GREEN,CmaesAgent:constants.BLUE}

        myrect = pygame.Rect((agentPosition[1]+1/3)*BLOCK_WIDTH, (agentPosition[0]+1/3)*BLOCK_HEIGHT, (BLOCK_WIDTH/3), (BLOCK_HEIGHT/3))
        myrect_full = pygame.Rect((agentPosition[1])*BLOCK_WIDTH, (agentPosition[0])*BLOCK_HEIGHT, (BLOCK_WIDTH), (BLOCK_HEIGHT))
        if agentPosition[2]==-1:
            pygame.draw.rect(surface, constants.GREY, myrect_full)
        else:
            pygame.draw.rect(surface, base_colors[agentPosition[2]], myrect_full)
        pygame.draw.rect(surface, agent_color_dict[type(agent)], myrect)

def initialize_game():
    pygame.init()
    surface = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
    pygame.display.set_caption(constants.TITLE)
    surface.fill(constants.UGLY_PINK)
    return surface


if __name__=="__main__":
    pass