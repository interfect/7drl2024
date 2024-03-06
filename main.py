#!/usr/bin/env python3

import tcod
import tcod.console
import tcod.context
import tcod.event
import tcod.tileset

from tcod.event import KeySym

from llama_cpp import Llama, LlamaGrammar

import json
import os
import random

from typing import Optional
from urllib.request import urlretrieve

class Generator:
    
    MODEL_URLS = {
        "tinyllama-1.1b-1t-openorca.Q4_K_M.gguf": "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf",
        "mistral-7b-v0.1.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf"
    }

    MODEL="tinyllama-1.1b-1t-openorca.Q4_K_M.gguf"
    #MODEL="mistral-7b-v0.1.Q4_K_M.gguf"

    def __init__(self):
        self.grammar = LlamaGrammar.from_string(open("./rogue.gbnf").read())

        if not os.path.exists(self.MODEL):
            print("Download model")
            urlretrieve(self.MODEL_URLS[self.MODEL], self.MODEL)
    
        print("Load model")
        self.model = Llama(
            model_path=self.MODEL,
        )
        print("Model loaded")
        
        
    def invent(self, prompt: str) -> dict:
        result = self.model(prompt, grammar=self.grammar, stop=["\n\n"], max_tokens=-1, mirostat_mode=2)
        generated = json.loads(result["choices"][0]["text"])
        print(f"Invented: {generated}")
        return generated
    
    def invent_object(self) -> dict:
        example = random.choice([
            "boring wall",
            "portcullis",
            "moderately large stick",
            "hole in the ground",
            "crushing sense of dread"
        ])
        return self.invent(f"Here is a JSON object describing an obstacle for my 7DRL roguelike, way better than \"{example}\":\n")
    
    def invent_enemy(self) -> dict:
        example = random.choice([
            "dire cat",
            "evil stick",
            "boss",
            "really cool guy who doesn't affraid of anything",
            "nasal demon"
        ])
        return self.invent(f"Here is a JSON object describing an enemy for my 7DRL roguelike, way better than \"{example}\":\n")

class GameState:
    """
    Game state base class.
    
    Can be swapped between for a game state state machine.
    """
    
    def render_to(self, console: tcod.console.Console) -> None:
        """
        Render this state to the given console.
        """
        raise NotImplementedError()
    
    def handle_event(self, event: tcod.event.Event) -> None:
        """
        Handle the given user input event.
        """
        raise NotImplementedError()
        
        
class WorldObject:
    """
    Represents an object in the world.
    """
    def __init__(self, x: int, y: int, symbol: str, name: str = "object", z_layer: int = 0) -> None:
        self.x = x
        self.y = y
        self.symbol = symbol
        self.name = name
        self.z_layer = z_layer
        
    def definite_name(self) -> str:
        """
        Get the name of the object formatted with a definite article, if applicable.
        """
        return f"the {self.name}"
    
    def indefinite_name(self) -> str:
        """
        Get the name of the object formatted with an indefinite article, if applicable.
        """
        VOWELS = "aeiou"
        if self.name.lower()[0] in VOWELS:
            article = "an"
        else:
            article = "a"
        return f"{article} {self.name}"
        
    def nominative_pronoun(self) -> str:
        """
        Get the nominative pronoun to refer to the object.
        """
        
        return "it"
        
class Enemy(WorldObject):
    """
    Represents an enemy that can be attacked.
    """
    def __init__(self, x: int, y: int, symbol: str, name: str = "enemy", health: int = 10):
        super().__init__(x, y, symbol, name)
        
        self.max_health = health
        self.health = health
         
class Player(WorldObject):
    def __init__(self) -> None:
        super().__init__(0, 0, "@", "Player", z_layer=1)
        
class PlayingState(GameState):
    """
    State for playing the game.
    
    Walk around as an @.
    """
    
    def __init__(self) -> None:
        """
        Set up a fresh game state.
        """
        
        # Set to true to trigger victory screen
        self.game_won = False
        
        # Hang on to the player specifically
        self.player = Player()
        
        # But put them in the list of all point objects.
        self.objects = [self.player]
        
        # Keep track of log messages and their counts
        self.logs: list[tuple[str, int]] = []
        
        # Make an initial map
        generator = Generator()
        MAP_RANGE = 10
        object_count = 0
        desired_object_count = random.randint(5, 7)
        while object_count < desired_object_count:
            x = random.randint(-MAP_RANGE, MAP_RANGE)
            y = random.randint(-MAP_RANGE, MAP_RANGE)
            if self.object_at(x, y) is None:
                # Put something here
                object_type = generator.invent_object()
                self.objects.append(WorldObject(x, y, object_type["symbol"], object_type["name"]))
                object_count += 1
        
        enemy_count = 0
        desired_enemy_count = random.randint(2, 5)
        while enemy_count < desired_enemy_count:
            x = random.randint(-MAP_RANGE, MAP_RANGE)
            y = random.randint(-MAP_RANGE, MAP_RANGE)
            if self.object_at(x, y) is None:
                enemy_type = generator.invent_enemy()
                self.objects.append(Enemy(x, y, enemy_type["symbol"], enemy_type["name"], int(enemy_type["health"])))
                enemy_count += 1
                
                
        self.log("Hello World")
            
    def object_at(self, x: int, y: int) -> Optional[WorldObject]:
        """
        Get the object at the given coordinates, or None.
        """
        for obj in self.objects:
            if obj.x == x and obj.y == y:
                return obj
        return None
        
    def log(self, message: str) -> None:
        if len(self.logs) > 0 and self.logs[-1][0] == message:
            # A duplicate. Increase the count.
            self.logs[-1] = (self.logs[-1][0], self.logs[-1][1] + 1)
        else:
            self.logs.append((message, 1))
    
    def render_to(self, console: tcod.console.Console) -> None:
        # Make sure higher-Z objects draw on top
        self.objects.sort(key=lambda o: o.z_layer)
        
        # Compute layout
        LOG_HEIGHT = 3
    
        console.clear()
        console.draw_frame(0, 0, console.width, console.height - LOG_HEIGHT, "Super RPG 640x480")
        
        # Draw a world view inset in the frame
        self.draw_world(console, 1, 1, console.width - 2, console.height - LOG_HEIGHT - 2)
        
        # Draw the log messages
        log_start_height = console.height - LOG_HEIGHT
        for log_message, log_count in reversed(self.logs):
            # Lay out log messages newest at the top, older below
            if log_count > 1:
                # Duplicate messages are expressed with counts
                log_message += f" x{log_count}"
            log_start_height += console.print_box(0, log_start_height, console.width, LOG_HEIGHT, log_message)
            if log_start_height >= console.height:
                break
                
        BANNER_WIDTH = 20
        BANNER_HEIGHT = 3
        if self.game_won:
            # Print a big victory banner
            console.draw_frame(console.width // 2 - BANNER_WIDTH // 2, console.height // 2 - BANNER_HEIGHT // 2, BANNER_WIDTH, BANNER_HEIGHT, decoration="╔═╗║ ║╚═╝", fg=(0, 255, 0))
            console.print_box(console.width // 2 - BANNER_WIDTH // 2 + 1, console.height // 2 - BANNER_HEIGHT // 2 + 1, BANNER_WIDTH - 2, BANNER_HEIGHT - 2, "You Win!", fg=(0, 255, 0), alignment=tcod.CENTER)
            
        
    def draw_world(self, console: tcod.console.Console, x: int, y: int, width: int, height: int) -> None:
        """
        Draw the world into a region of the given console.
        """
        
        # Find where to put the view upper left corner to center the player
        view_x = self.player.x - width // 2
        view_y = self.player.y - height // 2
        
        for to_render in self.objects:
            x_in_view = to_render.x - view_x
            y_in_view = to_render.y - view_y
            if x_in_view >= 0 and x_in_view < width and y_in_view >= 0 and y_in_view < height:
                console.print(x_in_view + x, y_in_view + y, to_render.symbol)

    DIRECTION_KEYS = {
        # Arrow keys
        KeySym.LEFT: (-1, 0),
        KeySym.RIGHT: (1, 0),
        KeySym.UP: (0, -1),
        KeySym.DOWN: (0, 1),
        # Arrow key diagonals
        KeySym.HOME: (-1, -1),
        KeySym.END: (-1, 1),
        KeySym.PAGEUP: (1, -1),
        KeySym.PAGEDOWN: (1, 1),
        # Keypad
        KeySym.KP_4: (-1, 0),
        KeySym.KP_6: (1, 0),
        KeySym.KP_8: (0, -1),
        KeySym.KP_2: (0, 1),
        KeySym.KP_7: (-1, -1),
        KeySym.KP_1: (-1, 1),
        KeySym.KP_9: (1, -1),
        KeySym.KP_3: (1, 1),
        # VI keys
        KeySym.h: (-1, 0),
        KeySym.l: (1, 0),
        KeySym.k: (0, -1),
        KeySym.j: (0, 1),
        KeySym.y: (-1, -1),
        KeySym.b: (-1, 1),
        KeySym.u: (1, -1),
        KeySym.n: (1, 1),
    }
    
    def handle_event(self, event: tcod.event.Event) -> None:
        if isinstance(event, tcod.event.KeyDown) and event.sym in self.DIRECTION_KEYS:
            if not self.game_won:
                # The player wants to move.
                direction = self.DIRECTION_KEYS[event.sym]
                
                next_x = self.player.x + direction[0]
                next_y = self.player.y + direction[1]
                
                obstruction = self.object_at(next_x, next_y)
                if obstruction is None:
                    # You can just move there
                    self.player.x = next_x
                    self.player.y = next_y
                elif isinstance(obstruction, Enemy):
                    # Time to fight!
                    damage = random.randint(1, 10)
                    obstruction.health -= damage
                    message = f"You attack {obstruction.definite_name()} for {damage} damage!"
                    if obstruction.health > 0:
                        message += f" Now {obstruction.nominative_pronoun()} has {obstruction.health}/{obstruction.max_health} HP."
                    else:
                        # It is dead now
                        self.objects.remove(obstruction)
                        message += f" You kill {obstruction.nominative_pronoun()}!"
                    self.log(message)
                    
                    # Check for winning
                    has_enemies = False
                    for obj in self.objects:
                        if isinstance(obj, Enemy):
                            has_enemies = True
                            break
                    if not has_enemies:
                        self.game_won = True
                        self.log("All enemies have been defeated! You are victorious!")
                else:
                    # The playeer is bumping something.
                    self.log(f"Your path is obstructed by {obstruction.indefinite_name()}!")
        

def force_min_size(context: tcod.context.Context) -> None:
    """
    Force the window to be at least a minimum size.
    """
    MIN_WINDOW_WIDTH = 640
    MIN_WINDOW_HEIGHT = 480
    context.sdl_window.size = (max(context.sdl_window.size[0], MIN_WINDOW_WIDTH), max(context.sdl_window.size[1], MIN_WINDOW_HEIGHT))

def main() -> None:
    #FONT="Alloy_curses_12x12.png"
    FONT="Curses_square_24.png"
    tileset = tcod.tileset.load_tilesheet(
        FONT, columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
    )
    tcod.tileset.procedural_block_elements(tileset=tileset)
    
    state = PlayingState()
    
    with tcod.context.new(tileset=tileset) as context:
        force_min_size(context)
        width, height = context.recommended_console_size()
        console = tcod.console.Console(width, height)
        
        last_window_size = context.sdl_window.size
        
        while True: 
            # Main loop
            
            # Render the current game state
            state.render_to(console)
            
            # Show that
            context.present(console, keep_aspect=True)
            
            # Handle events
            for event in tcod.event.wait():
                if isinstance(event, tcod.event.Quit):
                    raise SystemExit()
                elif isinstance(event, tcod.event.WindowResized):
                    if context.sdl_window.size != last_window_size:
                        force_min_size(context)
                        width, height = context.recommended_console_size()
                        console = tcod.console.Console(width, height)
                        last_window_size = context.sdl_window.size
                else:
                    # Other events are probably input so let the game state deal with them.
                    state.handle_event(event)
                    

if __name__ == "__main__":
    main()

