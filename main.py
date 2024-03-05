import tcod.console
import tcod.context
import tcod.event
import tcod.tileset

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
    
    def handle_event(self, event: tcod.event.Event):
        """
        Handle the given user input event.
        """
        raise NotImplementedError()
        
        
class Player:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.symbol = "@"
        
class PlayingState(GameState):
    """
    State for playing the game.
    
    Walk around as an @.
    """
    
    def __init__(self):
        self.player = Player()
    
    def render_to(self, console: tcod.console.Console):
        console.clear()
        console.draw_frame(0, 0, console.width, console.height, "Super RPG 640x480")
        console.print(1, 1, "Hello World")
        
        # Find where to put the view upper left corner to center the player
        view_x = self.player.x - console.width // 2
        view_y = self.player.y - console.height // 2
        
        for to_render in [self.player]:
            console.print(to_render.x - view_x, to_render.y - view_y, to_render.symbol)
        
    def handle_event(self, event: tcod.event.Event):
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

