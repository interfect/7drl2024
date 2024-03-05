import tcod.console
import tcod.context
import tcod.event
import tcod.tileset

from tcod.event import KeySym

import random

from typing import Optional

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
    def __init__(self, x: int, y: int, symbol: str, z_layer: int = 0) -> None:
        self.x = x
        self.y = y
        self.symbol = symbol
        self.z_layer = z_layer
        
class Player(WorldObject):
    def __init__(self) -> None:
        super().__init__(0, 0, "@", 1)
        
class PlayingState(GameState):
    """
    State for playing the game.
    
    Walk around as an @.
    """
    
    def __init__(self) -> None:
        self.player = Player()
        self.objects = [self.player]
        
        for _ in range(random.randrange(5, 10)):
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if self.object_at(x, y) is None:
                self.objects.append(WorldObject(x, y, "?"))
            
    def object_at(self, x: int, y: int) -> Optional[WorldObject]:
        """
        Get the object at the given coordinates, or None.
        """
        for obj in self.objects:
            if obj.x == x and obj.y == y:
                return obj
        return None
    
    def render_to(self, console: tcod.console.Console) -> None:
        # Make sure higher-Z objects draw on top
        self.objects.sort(key=lambda o: o.z_layer)
    
        console.clear()
        console.draw_frame(0, 0, console.width, console.height, "Super RPG 640x480")
        
        # Draw a world view inset in the frame
        self.draw_world(console, 1, 1, console.width - 2, console.height - 2)
        
        console.print(1, 1, "Hello World")
        
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
            # The player wants to move.
            direction = self.DIRECTION_KEYS[event.sym]
            
            next_x = self.player.x + direction[0]
            next_y = self.player.y + direction[1]
            
            obstruction = self.object_at(next_x, next_y)
            if obstruction is None:
                # You can just move there
                self.player.x = next_x
                self.player.y = next_y
            else:
                # The playeer is bumping something. Use/fight/take it.
                pass
            
        


def force_min_size(context: tcod.context.Context) -> None:
    """
    Force the window to be at least a minimum size.
    """
    MIN_WINDOW_WIDTH = 640
    MIN_WINDOW_HEIGHT = 480
    context.sdl_window.size = (max(context.sdl_window.size[0], MIN_WINDOW_WIDTH), max(context.sdl_window.size[1], MIN_WINDOW_HEIGHT))

def main() -> None:
    tileset = tcod.tileset.load_tilesheet(
        "Alloy_curses_12x12.png", columns=16, rows=16, charmap=tcod.tileset.CHARMAP_CP437
    )
    tcod.tileset.procedural_block_elements(tileset=tileset)
    
    state = PlayingState()
    
    with tcod.context.new(tileset=tileset) as context:
        force_min_size(context)
        width, height = context.recommended_console_size()
        console = tcod.console.Console(width, height)
        
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
                    force_min_size(context)
                    width, height = context.recommended_console_size()
                    console = tcod.console.Console(width, height)
                else:
                    # Other events are probably input so let the game state deal with them.
                    state.handle_event(event)
                    

if __name__ == "__main__":
    main()

