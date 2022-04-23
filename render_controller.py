import constants
import pygame

class RenderController(object):

    def __init__(self,num_blocks_w=12,num_blocks_h=12):
        pygame.init()
        self.num_blocks_w=num_blocks_w
        self.num_blocks_h=num_blocks_h
        self.surface = pygame.display.set_mode((constants.SCREEN_WIDTH, constants.SCREEN_HEIGHT))
        pygame.display.set_caption(constants.TITLE)
        self.surface.fill(constants.UGLY_PINK)
        self.BLOCK_WIDTH=constants.SCREEN_WIDTH/num_blocks_w
        self.BLOCK_HEIGHT=constants.SCREEN_HEIGHT/num_blocks_h

    def draw_map(self, map_tiles):
        for j, tile in enumerate(map_tiles):
            for i, tile_contents in enumerate(tile):
                myrect = pygame.Rect(i*self.BLOCK_WIDTH, j*self.BLOCK_HEIGHT, self.BLOCK_WIDTH, self.BLOCK_HEIGHT)
                pygame.draw.rect(self.surface, tile_contents, myrect)

    def draw_grid(self):
        for i in range(self.num_blocks_w):
            new_height = round(i * self.BLOCK_HEIGHT)
            new_width = round(i * self.BLOCK_WIDTH)
            pygame.draw.line(self.surface, constants.BLACK, (0, new_height), (constants.SCREEN_WIDTH, new_height), 2)
            pygame.draw.line(self.surface, constants.BLACK, (new_width, 0), (new_width, constants.SCREEN_HEIGHT), 2)

    def draw_text(self, text):
        font = pygame.font.SysFont(None, 12)
        img = font.render(text, True, constants.RED)
        self.surface.blit(img,(5,5))

    def draw_text_multiple_agents(self, agents):
        font = pygame.font.SysFont(None, 24)
        agent_color_dict={'HeuristicAgent':constants.RED,'ReinforcedAgent':constants.GREEN,'CmaesAgent':constants.BLUE}
        for e,agent in enumerate(agents):
            color=agent_color_dict[type(agent).__name__]
            img = font.render(f"POINTS COLLECTED: {agent.env.points_collected}", True, color)
            self.surface.blit(img,(5,15*e))

    def draw_agent(self, agentPosition):
        myrect = pygame.Rect((agentPosition[1]+1/3)*self.BLOCK_WIDTH, (agentPosition[0]+1/3)*self.BLOCK_HEIGHT, (self.BLOCK_WIDTH/3), (self.BLOCK_HEIGHT/3))
        pygame.draw.rect(self.surface, (255,255,255), myrect)

    def draw_bases(self, basesPosition, baseColor):
        for e,base in enumerate(basesPosition):
            myrect = pygame.Rect((base[1]+1/3)*self.BLOCK_WIDTH, (base[0]+1/3)*self.BLOCK_HEIGHT, (self.BLOCK_WIDTH/3), (self.BLOCK_HEIGHT/3))
            myrect_full = pygame.Rect((base[1])*self.BLOCK_WIDTH, (base[0])*self.BLOCK_HEIGHT, (self.BLOCK_WIDTH), (self.BLOCK_HEIGHT))
            pygame.draw.rect(self.surface, baseColor[e], myrect_full)
            pygame.draw.rect(self.surface, (0,0,0), myrect)

    def draw_bases_multiple_agents(self, basesPosition, baseColor):
        for e,base in enumerate(basesPosition):
            myrect = pygame.Rect((base[1]+1/3)*self.BLOCK_WIDTH, (base[0]+1/3)*self.BLOCK_HEIGHT, (self.BLOCK_WIDTH/3), (self.BLOCK_HEIGHT/3))
            myrect_full = pygame.Rect((base[1])*self.BLOCK_WIDTH, (base[0])*self.BLOCK_HEIGHT, (self.BLOCK_WIDTH), (self.BLOCK_HEIGHT))
            pygame.draw.rect(self.surface, baseColor[e], myrect_full)
            pygame.draw.rect(self.surface, (0,0,0), myrect)

    def draw_food_multiple_agents(self, food_list, baseColor):
        for food in food_list:
            myrect = pygame.Rect((food[1])*self.BLOCK_WIDTH, (food[0])*self.BLOCK_HEIGHT, (self.BLOCK_WIDTH), (self.BLOCK_HEIGHT))
            pygame.draw.rect(self.surface, baseColor[food[2]], myrect)


    def draw_multiple_agents(self,agents):
        for agent in agents:
            agentPosition=agent.env.agentPosition
            base_colors=agent.env.base_colors
            agent_color_dict={'HeuristicAgent':constants.RED,'ReinforcedAgent':constants.GREEN,'CmaesAgent':constants.BLUE}

            myrect = pygame.Rect((agentPosition[1]+1/3)*self.BLOCK_WIDTH, (agentPosition[0]+1/3)*self.BLOCK_HEIGHT, (self.BLOCK_WIDTH/3), (self.BLOCK_HEIGHT/3))
            myrect_full = pygame.Rect((agentPosition[1])*self.BLOCK_WIDTH, (agentPosition[0])*self.BLOCK_HEIGHT, (self.BLOCK_WIDTH), (self.BLOCK_HEIGHT))
            if agentPosition[2]==-1:
                pygame.draw.rect(self.surface, constants.GREY, myrect_full)
            else:
                pygame.draw.rect(self.surface, base_colors[agentPosition[2]], myrect_full)
            pygame.draw.rect(self.surface, agent_color_dict[type(agent).__name__], myrect)


if __name__=="__main__":
    pass