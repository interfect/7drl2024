import tcod.console
import tcod.context
import tcod.event
import tcod.tileset

def force_min_size(context: tcod.context.Context):
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
    
    with tcod.context.new(tileset=tileset) as context:
        force_min_size(context)
        width, height = context.recommended_console_size()
        console = tcod.console.Console(width, height)
        
        while True: 
            console.draw_frame(0, 0, console.width, console.height, "Super RPG 640x480")
            console.print(1, 1, "Hello World")
            
            context.present(console, keep_aspect=True)
            
            for event in tcod.event.wait():
                if isinstance(event, tcod.event.Quit):
                    raise SystemExit()
                elif isinstance(event, tcod.event.WindowResized):
                    force_min_size(context)
                    width, height = context.recommended_console_size()
                    console = tcod.console.Console(width, height)
                    

if __name__ == "__main__":
    main()

