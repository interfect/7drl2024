#!/usr/bin/env python3

import tcod.libtcodpy
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

    #MODEL="tinyllama-1.1b-1t-openorca.Q4_K_M.gguf"
    MODEL="mistral-7b-v0.1.Q4_K_M.gguf"
    
    # This holds a prompt template for each type of object.
    PROMPTS = {
        "enemy": "Here is a JSON object describing an enemy for my 7DRL roguelike, way better than \"{}\" from the last game:",
        "obstacle": "Here is a JSON object describing an obstacle for my 7DRL roguelike, way better than \"{}\" from the last game:",
        "loot": "Here is a JSON object describing a loot item for my 7DRL roguelike, way better than \"{}\" from the last game:"
    }
    
    # This holds example item types used to vary the prompt.
    EXAMPLES = {
        "enemy": [
            "dire cat",
            "evil stick",
            "boss",
            "really cool guy who doesn't affraid of anything",
            "nasal demon"
        ],
        "obstacle": [
            "boring wall",
            "portcullis",
            "moderately large stick",
            "hole in the ground"
        ],
        "loot": [
            "Excalibur",
            "pointy rock",
            "small pile of gold",
            "portable hole"
        ]
    }

    def __init__(self) -> None:
        """
        Make a new generator, which is responsible for remembering and producing object types.
        """
        
        # We lazily load special grammars for each type of object
        self.grammars: dict[str, LlamaGrammar] = {}
        # And we lazily load the model to generate new things
        self.model: Optional[Llama] = None
        
    def get_model(self) -> Llama:
        """
        Get the model to generate with.
        """
        if self.model is None:
            if not os.path.exists(self.MODEL):
                print("Download model")
                urlretrieve(self.MODEL_URLS[self.MODEL], self.MODEL)
        
            print("Load model")
            self.model = Llama(
                model_path=self.MODEL,
            )
        return self.model
        
    def get_grammar(self, object_type: str) -> LlamaGrammar:
        """
        Get the grammar to use to generate the given type of object.
        """
        if object_type not in self.grammars:
            object_grammar = open(os.path.join("grammars", f"{object_type}.gbnf")).read()
            common_grammar = open(os.path.join("grammars", "common.gbnf")).read()
            self.grammars[object_type] = LlamaGrammar.from_string("\n".join([object_grammar, common_grammar]))
        return self.grammars[object_type]
    
    def invent_object(self, object_type: str, **features) -> dict:
        """
        Generate a new, never-before-seen object of the given type, with the given parameters.
        """
        
        # Get a prompt with a random example in it
        prompt = self.PROMPTS[object_type].format(random.choice(self.EXAMPLES[object_type]))
        
        # Add any existing keys, leaving off the closing brace and adding a trailing comma
        prompt += "\n\n```\n" + json.dumps(features, indent=2)[:-1].rstrip() + "," if len(features) > 0 else ""
        
        # Run the model
        result = self.get_model()(prompt, grammar=self.get_grammar(object_type), stop=["\n\n"], max_tokens=-1, mirostat_mode=2)
        
        # Load up whatever keys it came up with
        new_keys = json.loads("{" + result["choices"][0]["text"])
        
        # Combine the keys together
        obj = dict(features)
        obj.update(new_keys)
        print(f"Invented: {obj}")
        return obj
        
        
    def select_object(self, object_type: str) -> dict:
        """
        Get a dict defining an object of the given type.
        
        All types have "symbol", "name", "definite_article", and "indefinite_article".
        
        "enemy": additionally has "health"
        
        "obstacle": has no additional fields. 
        """
        
        # We maintain a database of objects by type and rarity
        rarity = random.choice(["common"] * 10 + ["uncomon"] * 3 + ["rare"])
        
        # Within each we have 10 types
        type_num = random.randrange(0, 10)
        
        # Where should that file be?
        path = os.path.join("objects", object_type, rarity, f"{type_num}.json")
        if not os.path.exists(path):
            # Make directory
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Invent the object type
            obj = self.invent_object(object_type, rarity=rarity)
            # And save it
            json.dump(obj, open(path, 'w'))
        
        return json.load(open(path))

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
    
    NOMINATIVE_TO_ACCUSATIVE = {
        "it": "it",
        "he": "him",
        "she": "her",
        "they": "them"
    }
    
    HAS_HAVE = {
        "it": "has",
        "he": "has",
        "she": "has",
        "they": "have"
    }
    
    IS_ARE = {
        "it": "is",
        "he": "is",
        "she": "is",
        "they": "are"
    }
    
    def __init__(
        self,
        x: int,
        y: int,
        symbol: str = "?",
        color: str = "#ffffff",
        name: str = "object",
        indefinite_article: Optional[str] = None,
        definite_article: Optional[str] = None,
        nominative_pronoun: str = "it",
        rarity: str = "common",
        z_layer: int = 0
    ) -> None:
        
        self.x = x
        self.y = y
        self.symbol = symbol
        self.fg = self.hex_to_rgb(color)
        self.name = name
        self.indefinite_article = indefinite_article
        self.definite_article = definite_article
        self.nominative_pronoun = nominative_pronoun
        self.accusative_pronoun = self.NOMINATIVE_TO_ACCUSATIVE[nominative_pronoun]
        self.has_have = self.HAS_HAVE[nominative_pronoun]
        self.is_are = self.IS_ARE[nominative_pronoun]
        self.rarity = rarity
        self.z_layer = z_layer
    
    def hex_to_rgb(self, hex_code: str) -> tuple[int, int, int]:
        """
        Convert a hex color code with leading # to an RGB tuple out of 255.
        
        See <https://stackoverflow.com/a/71804445>
        """
        return tuple(int(hex_code[i:i+2], 16)  for i in (1, 3, 5))
    
    def definite_name(self) -> str:
        """
        Get the name of the object formatted with a definite article, if applicable.
        """
        parts = []
        if self.definite_article:
            parts.append(self.definite_article)
        parts.append(self.name)
        return " ".join(parts)
    
    def indefinite_name(self) -> str:
        """
        Get the name of the object formatted with an indefinite article, if applicable.
        """
        
        parts = []
        if self.indefinite_article:
            parts.append(self.indefinite_article)
        parts.append(self.name)
        return " ".join(parts)
        
        
class Enemy(WorldObject):
    """
    Represents an enemy that can be attacked.
    """
    def __init__(self, x: int, y: int, health: int = 10, **kwargs):
        super().__init__(x, y, **kwargs)
        
        self.max_health = health
        self.health = health
        
        # An enemy can be carrying loot items
        self.inventory: list[WorldObject] = []
         
class Player(WorldObject):
    def __init__(self) -> None:
        super().__init__(0, 0, symbol="@", name="Player", z_layer=1)
        
        # We collect loot items from enemies
        self.inventory: list[WorldObject] = []
        
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
                object_type = generator.select_object("obstacle")
                self.objects.append(WorldObject(x, y, **object_type))
                object_count += 1
        
        enemy_count = 0
        desired_enemy_count = random.randint(2, 5)
        while enemy_count < desired_enemy_count:
            x = random.randint(-MAP_RANGE, MAP_RANGE)
            y = random.randint(-MAP_RANGE, MAP_RANGE)
            if self.object_at(x, y) is None:
                enemy_type = generator.select_object("enemy")
                enemy = Enemy(x, y, **enemy_type)
                enemy.inventory.append(WorldObject(0, 0, **generator.select_object("loot")))
                self.objects.append(enemy)
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
        LOG_HEIGHT = 4
    
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
            console.print(0, log_start_height, "•")
            log_start_height += console.print_box(1, log_start_height, console.width, LOG_HEIGHT, log_message)
            if log_start_height >= console.height:
                break
                
        BANNER_WIDTH = 20
        BANNER_HEIGHT = 3
        if self.game_won:
            # Print a big victory banner
            console.draw_frame(console.width // 2 - BANNER_WIDTH // 2, console.height // 2 - BANNER_HEIGHT // 2, BANNER_WIDTH, BANNER_HEIGHT, decoration="╔═╗║ ║╚═╝", fg=(0, 255, 0))
            console.print_box(console.width // 2 - BANNER_WIDTH // 2 + 1, console.height // 2 - BANNER_HEIGHT // 2 + 1, BANNER_WIDTH - 2, BANNER_HEIGHT - 2, "You Win!", fg=(0, 255, 0), alignment=tcod.libtcodpy.CENTER)
            
        
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
                console.print(x_in_view + x, y_in_view + y, to_render.symbol, fg=to_render.fg)

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
                    hit_message = f"You attack {obstruction.definite_name()} for {damage} damage!"
                    if obstruction.health > 0:
                        hit_message += f" Now {obstruction.nominative_pronoun} {obstruction.has_have} {obstruction.health}/{obstruction.max_health} HP."
                        self.log(hit_message)
                    else:
                        # It is dead now
                        self.objects.remove(obstruction)
                        hit_message += f" You kill {obstruction.accusative_pronoun}!"
                        self.log(hit_message)
                        
                        # Take its stuff
                        loot = random.choice(obstruction.inventory) if len(obstruction.inventory) > 0 else None
                        if loot is not None:
                            self.player.inventory.append(loot)
                            rarity_article = "a" if loot.rarity != "uncomon" else "an"
                            self.log(f"You loot {loot.indefinite_name()}, {rarity_article} {loot.rarity} treasure.")
                    
                    # Check for winning
                    has_enemies = False
                    for obj in self.objects:
                        if isinstance(obj, Enemy):
                            has_enemies = True
                            break
                    if not has_enemies:
                        self.game_won = True
                        self.log("All enemies have been defeated!")
                else:
                    # The playeer is bumping something.
                    self.log(f"Your path is obstructed by {obstruction.indefinite_name()}!")
        if isinstance(event, tcod.event.KeyDown) and event.sym == tcod.event.KeySym.ESCAPE:
            if self.game_won:
                # Let the user quit at the end
                raise SystemExit()
        

def force_min_size(context: tcod.context.Context) -> None:
    """
    Force the window to be at least a minimum size.
    """
    MIN_WINDOW_WIDTH = 640
    MIN_WINDOW_HEIGHT = 480
    context.sdl_window.size = (max(context.sdl_window.size[0], MIN_WINDOW_WIDTH), max(context.sdl_window.size[1], MIN_WINDOW_HEIGHT))
    
def force_normal_shape(context: tcod.context.Context) -> None:
    """
    Force the window to be a sensible shape.
    """
    # Don't make it skinnier than 2 to 1 in any dimension
    if context.sdl_window.size[0] / context.sdl_window.size[1] > 2:
        context.sdl_window.size = (2 * context.sdl_window.size[1], context.sdl_window.size[1])
    if context.sdl_window.size[1] / context.sdl_window.size[0] > 2:
        context.sdl_window.size = (context.sdl_window.size[0], 2 * context.sdl_window.size[0])
        

def main() -> None:
    #FONT="Alloy_curses_12x12.png"
    FONT="Curses_square_24.png"
    tileset = tcod.tileset.load_tilesheet(
        FONT, columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
    )
    tcod.tileset.procedural_block_elements(tileset=tileset)
    
    state = PlayingState()
    
    with tcod.context.new(tileset=tileset) as context:
        force_normal_shape(context)
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

