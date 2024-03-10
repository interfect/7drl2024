#!/usr/bin/env python3

import tcod.bsp
import tcod.console
import tcod.context
import tcod.event
import tcod.image
import tcod.libtcodpy
import tcod.tileset

from tcod.event import KeySym

from llama_cpp import Llama, LlamaGrammar

import json
import math
import os
import sys
import types
import random

from enum import IntEnum
from contextlib import contextmanager
from queue import Queue, Empty
from threading import Thread, Lock
from typing import Optional, Generator
from urllib.request import urlopen

GAME_NAME = "False Ghost"

class ProceduralGenerator:
    
    MODEL_URLS = {
        "tinyllama-1.1b-1t-openorca.Q4_K_M.gguf": "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_K_M.gguf",
        "mistral-7b-v0.1.Q4_K_M.gguf": "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_K_M.gguf"
    }

    #MODEL="tinyllama-1.1b-1t-openorca.Q4_K_M.gguf"
    MODEL="mistral-7b-v0.1.Q4_K_M.gguf"
    
    # What special characters could we use.
    # TODO: Actually use these. They can't be in the grammar. Try rolling them ourselves and showing the model?
    SPECIAL_CHARS="☺☻♥♦♣♠•◘○◙♂♀♪♫☼►◄↕‼¶§▬↨↑↓→←∟↔▲▼"
    
    # This holds a prompt template for each type of object.
    PROMPTS = {
        "enemy": "An enemy for my 7DRL roguelike, way better than \"{}\" or \"{}\" from the last game:",
        "obstacle": "An obstacle in a level for my 7DRL roguelike, way better than \"{}\" or \"{}\" from the last game:",
        "loot": "A loot item for my 7DRL roguelike, way better than \"{}\" or \"{}\" from the last game:"
    }
    
    # This holds example item types used to vary the prompt.
    EXAMPLES = {
        "enemy": [
            "dire cat",
            "evil stick",
            "boss",
            "CEO",
            "mecha-CEO",
            "really cool guy",
            "nasal demon",
            "tiger shark",
        ],
        "obstacle": [
            "boring wall",
            "portcullis",
            "moderately large stick",
            "hole in the ground",
            "missing floor tile",
            "pile of rusty metal",
        ],
        "loot": [
            "double-ended sword",
            "pointy rock",
            "small pile of gold",
            "portable hole",
            "Spear of Justice",
            "stapler",
            "gun",
        ]
    }
    
    # What elemental domains are there?
    ELEMENTAL_DOMAINS = ["normal", "earth", "air", "fire", "water", "good", "evil", "business"]
    
    # How likely is a thing to differ from the domain of its parent?
    DIFFERENT_DOMAIN_CHANCE = 0.2
    
    # Which domains beat which others?
    BEAT_PAIRS = {
        ("water", "fire"),
        ("fire", "air"),
        ("air", "earth"),
        ("earth", "water"),
        ("evil", "good"),
        ("good", "business"),
        ("business", "evil")
    }
    
    @classmethod
    def strong_against(cls, domain_a: str, domain_b: str) -> bool:
        """
        Return True if the first elemental domain should deal more damage than usual to the second.
        """
        
        return (domain_a, domain_b) in cls.BEAT_PAIRS
        
    @classmethod
    def weak_against(cls, domain_a: str, domain_b: str) -> bool:
        """
        Return True if the first elemental domain should deal less damage than usual to the second.
        """
        
        return cls.strong_against(domain_b, domain_a)
        

    def __init__(self) -> None:
        """
        Make a new generator, which is responsible for remembering and producing object types.
        """
        
        # We lazily load special grammars for each type of object
        self.grammars: dict[str, LlamaGrammar] = {}
        # And we lazily load the model to generate new things
        self.model: Optional[Llama] = None
    
    @types.coroutine
    def download_model(self) -> Generator[tuple[int, int, str], None, None]:
        """
        Download the model file we are meant to use.
        
        Doesn't download it twice.
        
        Yields progress events.
        """

        source_url = self.MODEL_URLS[self.MODEL]
        dest_path = self.MODEL
        
        MEGABYTE = 1024 * 1024
        
        if not os.path.exists(dest_path):
            with open(dest_path + ".tmp", "wb") as out_handle:
                yield (0, 0, f"Downloading {dest_path}")
                with urlopen(source_url) as handle:
                    expected_length = int(handle.headers.get('Content-Length'))
                    bytes_read = 0
                    while True:
                        yield (bytes_read // MEGABYTE, expected_length // MEGABYTE, f"Downloading {dest_path}")
                        buffer = handle.read(MEGABYTE)
                        if len(buffer) == 0:
                            # Hit EOF
                            break
                        bytes_read += len(buffer)
                        out_handle.write(buffer)
                    yield (bytes_read // MEGABYTE, expected_length // MEGABYTE, f"Downloading {dest_path}") 
            os.rename(dest_path + ".tmp", dest_path)             
    
    def get_model(self) -> Llama:
        """
        Get the model to generate with.
        """
        if self.model is None:
            if not os.path.exists(self.MODEL):
                print("Download model")
                task = self.download_model()
                for _ in task:
                    # Run the generator to completion
                    pass
        
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
        
        # Get a prompt with two random examples in it
        chosen_examples = random.sample(self.EXAMPLES[object_type], 2)
        prompt = self.PROMPTS[object_type].format(chosen_examples[0], chosen_examples[1])
        
        # Add any existing keys, leaving off the closing brace and adding a trailing comma
        prompt += "\n\n```\n" + json.dumps(features, indent=2)[:-1].rstrip() + "," if len(features) > 0 else ""
        
        # Run the model
        result = self.get_model()(prompt, grammar=self.get_grammar(object_type), stop=["\n\n"], max_tokens=-1, mirostat_mode=2)
        
        result_text = result["choices"][0]["text"]
        
        print(result_text)
        
        # Load up whatever keys it came up with
        new_keys = json.loads("{" + result_text)
        
        # Combine the keys together
        obj = dict(features)
        obj.update(new_keys)
        print(f"Invented: {obj}")
        return obj
        
    def select_rarity(self, level_number: int) -> str:
        """
        Select a rarity level for an item at the given dungeon level.
        """
        
        return random.choice(["common"] * 10 + ["uncommon"] * 3 + ["rare"] * level_number)
        
        
    def select_object(self, object_type: str, rarity: str, elemental_domain: Optional[str] = None) -> dict:
        """
        Get a dict defining an object of the given type and rarity.
        
        Rarity can be "common", "uncommon", or "rare".
        
        All types have:
            "symbol",
            "name",
            "definite_article",
            "indefinite_article",
            "nominative_pronoun",
            "elemental_domain"
        
        "enemy": additionally has "health".
        
        "obstacle": has no additional fields. 
        
        "loot": additionally has a "value".
        """

        if elemental_domain is None:
            # Pick an elemental domain if not provided
            elemental_domain = self.select_domain()
        
        # Within each we have 3 types
        type_num = random.randrange(0, 3)
        
        # Where should that file be?
        path = os.path.join("objects", object_type, elemental_domain, rarity, f"{type_num}.json")
        if not os.path.exists(path):
            # Make directory
            os.makedirs(os.path.dirname(path), exist_ok=True)

            # Invent the object type
            obj = self.invent_object(object_type, rarity=rarity, elemental_domain=elemental_domain)
            
            # And save it
            json.dump(obj, open(path, 'w'))
        
        return json.load(open(path))
        
    def select_domain(self, parent_domain: Optional[str] = None):
        """
        Pick an elemental domain, possibly given an elemental domain of the containing thing.
        """
        
        if parent_domain is None or random.random() < self.DIFFERENT_DOMAIN_CHANCE:
            # Don't match the parent
            return random.choice(self.ELEMENTAL_DOMAINS)
        else:
            # Match the parent
            return parent_domain

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
    
    def handle_event(self, event: Optional[tcod.event.Event]) -> Optional["GameState"]:
        """
        Handle the given user input event.
        
        Event can be None if we are calling this method because it hasn't been
        called in a while; it is also the tick method.
        
        Returns the next state, or None to keep the current state.
        """
        raise NotImplementedError()
        
    def get_wait(self) -> float:
        return 0.1
        
        
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
        elemental_domain: str = "normal",
        z_layer: int = 0
    ) -> None:
        
        self.x = x
        self.y = y
        self.symbol = symbol
        self.fg = self.hex_to_rgb(color)
        self.name = name
        if self.name is None:
            # Oops we rolled a none name with a bad grammar
            self.name = "(unnamed)"
        self.indefinite_article = indefinite_article
        if self.indefinite_article == "a" and self.name[0].lower() in "aeiou":
            # Fix article
            self.indefinite_article = "an"
        elif self.indefinite_article == "an" and self.name[0].lower() not in "aeiou":
            # Fix article the other way
            self.indefinite_article = "a"
        self.definite_article = definite_article
        self.nominative_pronoun = nominative_pronoun
        self.accusative_pronoun = self.NOMINATIVE_TO_ACCUSATIVE[nominative_pronoun]
        self.has_have = self.HAS_HAVE[nominative_pronoun]
        self.is_are = self.IS_ARE[nominative_pronoun]
        self.rarity = rarity
        self.elemental_domain = elemental_domain
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
        if self.indefinite_article is not None:
            parts.append(self.indefinite_article)
        parts.append(self.name)
        return " ".join(parts)
        
    def hit(self, other: "GameObject") -> tuple[int, str]:
        """
        Hit one game object with another.
        
        Return the damage that should be dealt, and an adverb describing the effectiveness, like "brutally" or "futilely", or an empty string for normal.
        """
        
        if isinstance(self, Player):
            # Unarmed players can't do much damage
            max_damage = 4
        else:
            # Rarer items get bigger dice
            max_damage = {
                "common": 8,
                "uncommon": 10,
                "rare": 12
            }[self.rarity]
        
        base_damage = random.randint(1, max_damage)
        
        if ProceduralGenerator.strong_against(self.elemental_domain, other.elemental_domain):
            return base_damage * 2, "brutally"
        elif ProceduralGenerator.weak_against(self.elemental_domain, other.elemental_domain):
            return base_damage // 4, "futilely"
        else:
            return base_damage, ""
            

class Loot(WorldObject):
    """
    Represents a lootable inventory item.
    """
    def __init__(self, x: int, y: int, value: int = 0, **kwargs) -> None:
        super().__init__(x, y, **kwargs)
        
        # Save the gold value
        self.value = value
        
class Enemy(WorldObject):
    """
    Represents an enemy that can be attacked.
    """
    def __init__(self, x: int, y: int, health: int = 10, **kwargs):
        super().__init__(x, y, **kwargs)
        
        self.max_health = health
        self.health = health
        
        # An enemy can be carrying loot items
        self.inventory: list[Loot] = []


       
class Player(WorldObject):
    def __init__(self) -> None:
        super().__init__(0, 0, symbol="@", name="Player", z_layer=1)
        
        # We collect loot items from enemies
        self.inventory: list[Loot] = []
        
        # We have a selected item index, or None for no item.
        self.held_item_index: Optional[int] = None
        
        # And we have health
        self.max_health = 200
        self.health = self.max_health
        
        # And we have a total value score for the items we got
        self.value = 0
        
    def acquire_loot(self, loot: Loot) -> None:
        """
        Add the given loot to the player's inventory.
        """
        
        self.inventory.append(loot)
        self.value += loot.value
        
    def get_held_item(self) -> Optional[Loot]:
        """
        Get the item the player is using.
        """
        if self.held_item_index is None:
            return None
        return self.inventory[self.held_item_index]
        
    def next_item(self) -> None:
        """
        Hold the next item.
        """
        
        if self.held_item_index is None:
            if len(self.inventory) > 0:
                # Pick first item
                self.held_item_index = 0
        elif self.held_item_index + 1 < len(self.inventory):
            # Pick next item
            self.held_item_index += 1
        else:
            # Pick no item
            self.held_item_index = None
            
    def previous_item(self) -> None:
        """
        Hold the previous item.
        """
        
        if self.held_item_index is None:
            if len(self.inventory) > 0:
                # Pick last item
                self.held_item_index = len(self.inventory) - 1
        elif self.held_item_index - 1 >= 0:
            # Pick previous item
            self.held_item_index -= 1
        else:
            # Pick no item
            self.held_item_index = None
            
        

    
class Terrain(IntEnum):
    VOID = 0
    FLOOR = 1
    WALL = 2
    
    @staticmethod
    def to_symbol(value: "Terrain") -> str:
        """
        Get the symbol to represent a kind of terrain.
        """
        return {
            Terrain.VOID: " ",
            Terrain.FLOOR: ".",
            Terrain.WALL: "#"
        }[value]

class GameWorld:
    """
    World of the game.
    
    Holds the player and also the other things on the level.
    """
    
    def __init__(self, level_number: int, player: Player) -> None:
        """
        Set up a fresh world, using the given player.
        """
        
        # Remember the level number
        self.level_number = level_number
        # And the domain which will get filled in when we generate it.
        self.level_domain: Optional[str] = None
        
        # Hang on to the player specifically
        self.player = player
        # Make sure the player starts at full health
        self.player.health = self.player.max_health
        
        # But put them in the list of all point objects.
        self.objects: List[WorldObject] = [self.player]
        
        # Holds a map of terrins
        self.terrain: list[list[Terrain]] = []
        
        # Holds a list of room x, y, width, height tuples
        self.rooms: list[tuple[int, int, int, int]] = []
    
    def clear_terrain(self, x: int, y: int) -> None:
        """
        Make a new empty terrain of the given size.
        """
        
        self.terrain = [[Terrain.VOID for _ in range(y)] for __ in range(x)]
        
    def get_map_width(self) -> int:
        """
        Get the width of the current map.
        """
        return len(self.terrain)
        
    def get_map_height(self) -> int:
        """
        Get the height of the current map.
        """
        return len(self.terrain[0]) if self.terrain else 0
        
    def set_terrain(self, x: int, y: int, value: Terrain, if_value: Optional[Terrain] = None) -> None:
        """
        Set a terrain cell to the given value.
        
        If if_value is set, only changes from the given terrain type.
        """
        if if_value is None or self.terrain[x][y] == if_value:
            self.terrain[x][y] = value
        
    def set_terrain_region(self, x: int, y: int, width: int, height: int, value: Terrain, if_value: Optional[Terrain] = None) -> None:
        """
        Set terrain in an area to the given type.
        
        If if_value is set, only changes from the given terrain type.
        """
        
        for i in range(x, x + width):
            for j in range (y, y + height):
                self.set_terrain(i, j, value, if_value=if_value)
                
    def set_terrain_walls(self, x: int, y: int, width: int, height: int, value: Terrain, if_value: Optional[Terrain] = None) -> None:
        """
        Set terrain around the edges of an area to the given type.
        
        If if_value is set, only changes from the given terrain type.
        """
        
        # Just draw 4 wall lines
        self.set_terrain_region(x, y, width, 1, value, if_value=if_value)
        self.set_terrain_region(x, y + height - 1, width, 1, value, if_value=if_value)
        self.set_terrain_region(x, y, 1, height, value, if_value=if_value)
        self.set_terrain_region(x + width - 1, y, 1, height, value, if_value=if_value)
        
        
    def terrain_at(self, x: int, y: int) -> Terrain:
        """
        Get the value of the terrain at a location, which may not be in the terrain bounds.
        """
        
        if x < 0 or x >= len(self.terrain):
            return Terrain.VOID
        if y < 0 or y >= len(self.terrain[x]):
            return Terrain.VOID
        return self.terrain[x][y]
        
    @types.coroutine
    def generate_map(self) -> Generator[tuple[int, int, str], None, None]:
        """
        Make a terrain map to paly on.
        
        Structured as a coroutine generator; level is done when it stops.
        
        It yields progress tuples of completed out of total, and progress message.
        """

        yield (0, 0, "Generating terrain")
        
        MAP_WIDTH = 32
        MAP_HEIGHT = 32
        MAP_LEVELS = 5
        ROOM_MIN_INTERIOR_SIZE = 3
        ROOM_CHANCE = 0.5
        self.clear_terrain(MAP_WIDTH, MAP_HEIGHT)
        self.rooms = []
        
        bsp = tcod.bsp.BSP(x=0, y=0, width=MAP_WIDTH, height=MAP_HEIGHT)
        bsp.split_recursive(
            depth=MAP_LEVELS,
            min_width=ROOM_MIN_INTERIOR_SIZE + 1,
            min_height=ROOM_MIN_INTERIOR_SIZE + 1,
            max_horizontal_ratio=1.5,
            max_vertical_ratio=1.5
        )
        
        node_total = len(list(bsp.post_order()))
        node_count = 0
        yield (node_count, node_total, "Generating terrain")

        for node in bsp.post_order():
            # Post-order visits the parent after the children, so we make rooms and then dig to connect them.
            if node.children:
                # Connect the two child rooms somehow
                node1, node2 = node.children
                center1 = (node1.x + node1.width // 2, node1.y + node1.height // 2)
                center2 = (node2.x + node2.width // 2, node2.y + node2.height // 2)
                if not node.horizontal:
                    # This node is not from a horizontal split itself, so its children will be.
                    # node1 is left of node2
                    self.set_terrain_walls(center1[0] - 1, center1[1] - 1, center2[0] - center1[0] + 2, 3, Terrain.WALL, if_value=Terrain.VOID)
                    self.set_terrain_region(center1[0], center1[1], center2[0] - center1[0], 1, Terrain.FLOOR)
                else:
                    # node1 is above node2
                    self.set_terrain_walls(center1[0] - 1, center1[1] - 1, 3, center2[1] - center1[1] + 2, Terrain.WALL, if_value=Terrain.VOID)
                    self.set_terrain_region(center1[0], center1[1], 1, center2[1] - center1[1], Terrain.FLOOR)
            else:
                # Maybe make a room out of this node.
                if random.random() < ROOM_CHANCE:
                    # Carve out this room
                    self.set_terrain_region(node.x + 1, node.y + 1, node.width - 2, node.height - 2, Terrain.FLOOR)
                    self.set_terrain_walls(node.x, node.y, node.width, node.height, Terrain.WALL)
                    
                    # And remember it to populate. 
                    self.rooms.append((node.x + 1, node.y + 1, node.width - 2, node.height - 2))
                # Otherwise leave it alone and just connect to its center
            node_count += 1
            yield (node_count, node_total, "Generating terrain")
            
    
    def free_spaces_in(self, x: int, y: int, width: int, height: int, count: int) -> Generator[tuple[int, int], None, None]:
        """
        Get the given number of free spaces in the given region.
        """
        
        found = 0
        missed = 0
        while found < count:
            chosen_x = random.randrange(x, x + width)
            chosen_y = random.randrange(y, y + height)
            if self.free_space_at(chosen_x, chosen_y):
                yield (chosen_x, chosen_y)
                found += 1
            else:
                missed += 1
                if missed > 100 * count:
                    raise RuntimeError(f"Extreme bad luck in {x}, {y}, {width}, {height}")
    
    @types.coroutine  
    def populate_room(self, x: int, y: int, width: int, height: int, generator: ProceduralGenerator) -> Generator[tuple[int, int, str], None, None]:
        """
        Place some objects of the gievn type in the given area.
        """
        
        print(f"Populate {x}, {y}, {width}, {height}")
        
        room_domain = generator.select_domain(self.level_domain)
        
        # We keep obstacles walls to stay out of the doors.
        free_spaces = (width - 2) * (height - 2)
        
        # Make obstacles
        object_count = 0
        desired_object_count = random.choice([0, 0, 1, 3])
        if free_spaces >= desired_object_count:
            yield (object_count, desired_object_count, "Making obstacles")
            for pos in self.free_spaces_in(x + 1, y + 1, width - 2, height - 2, desired_object_count):
                # Put something here
                object_type = generator.select_object(
                    "obstacle",
                    rarity=generator.select_rarity(self.level_number),
                    elemental_domain=generator.select_domain(room_domain)
                )
                self.add_object(WorldObject(pos[0], pos[1], **object_type))
                object_count += 1
                yield (object_count, desired_object_count, "Making obstacles")
        
        
        # Enemies can block doors
        free_spaces = width * height - object_count
        
        # Make enemies
        enemy_count = 0
        desired_enemy_count = random.choice([0] * 5 + [1] * 2 + [2] * self.level_number)
        if free_spaces >= desired_enemy_count:
            yield (enemy_count, desired_enemy_count, "Making enemies")
            for pos in self.free_spaces_in(x, y, width, height, desired_enemy_count):
                # Pick a rarity for the enemy and its item
                rarity = generator.select_rarity(self.level_number)
                # Make an enemy for probably the room's domain
                enemy_domain = generator.select_domain(room_domain)
                enemy_type = generator.select_object("enemy", rarity=rarity, elemental_domain=enemy_domain)
                if enemy_type["health"] > self.level_number * 20:
                    # Don't let it be too strong too soon
                    enemy_type["health"] = self.level_number * 20
                enemy = Enemy(pos[0], pos[1], **enemy_type)
                yield (enemy_count, desired_enemy_count, "Making enemies")
                # Make a loot for probably the enemy's domain
                loot_domain = generator.select_domain(enemy_domain)
                loot_type = generator.select_object("loot", rarity=rarity, elemental_domain=loot_domain)
                enemy.inventory.append(Loot(pos[0], pos[1], **loot_type))
                self.add_object(enemy)
                enemy_count += 1
                yield (enemy_count, desired_enemy_count, "Making enemies")
    
    @types.coroutine    
    def generate_level(self, generator: ProceduralGenerator) -> Generator[tuple[int, int, str], None, None]:
        """
        Make a level to play.
        
        Structured as a coroutine generator; level is done when it stops.
        
        It yields progress tuples of completed out of total, and progress message.
        """
        
        # Make sure the model is available
        yield from generator.download_model()
        
        # Throw out the old objects, except the player who is here
        self.objects = [self.player]
        
        # Pick an elemental domain
        self.level_domain = generator.select_domain()
        
        # Make some terrain
        yield from self.generate_map()
        
        for room_number, room in enumerate(self.rooms):
            # Put stuff in each room
            for (done, total, message) in self.populate_room(room[0], room[1], room[2], room[3], generator):
                yield (room_number, len(self.rooms), message + f" ({done +  1}/{total}) in room {room_number + 1}/{len(self.rooms)}")
        
        # Put the player somewhere
        for x, y in self.free_spaces_in(0, 0, self.get_map_width(), self.get_map_height(), 1):
            # Found a place for the player
            self.player.x = x
            self.player.y = y
            
        if not self.has_enemies():
            # Make sure we have at least one enemy somewhere.
            for pos in self.free_spaces_in(0, 0, self.get_map_width(), self.get_map_height(), 1):
                # Make just one rare enemy with nothing
                enemy_type = generator.select_object("enemy", rarity="rare", elemental_domain=self.level_domain)
                # Buff it!
                enemy_type["health"] *= 2
                enemy = Enemy(pos[0], pos[1], **enemy_type)
                self.add_object(enemy)
        

        
    
    def object_at(self, x: int, y: int) -> Optional[WorldObject]:
        """
        Get the object at the given coordinates, or None.
        """
        for obj in self.objects:
            if obj.x == x and obj.y == y:
                return obj
        return None
        
    def free_space_at(self, x: int, y: int) -> bool:
        """
        Return True if there is free space for an object at the given position.
        """
        
        return self.terrain_at(x, y) == Terrain.FLOOR and self.object_at(x, y) is None
        
    def has_enemies(self) -> bool:
        """
        Return True if any enemies are left.
        """
        for obj in self.objects:
            if isinstance(obj, Enemy):
                return True
        return False
        
    def remove_object(self, obj: WorldObject) -> None:
        """
        Remove the given object from the world.
        """
        
        self.objects.remove(obj)
        
    def add_object(self, obj: WorldObject) -> None:
        """
        Add the given object to the world.
        """
        
        self.objects.append(obj)
        
    def draw(self, console: tcod.console.Console, x: int, y: int, width: int, height: int) -> None:
        """
        Draw the world centere don the player into a region of the given console.
        """
        
        # Make sure higher-Z objects draw on top
        self.objects.sort(key=lambda o: o.z_layer)
        
        # Find where to put the view upper left corner to center the player
        view_x = self.player.x - width // 2
        view_y = self.player.y - height // 2
        
        # Draw the terrain
        for x_in_view in range(width):
            world_x = x_in_view + view_x
            for y_in_view in range(height):
                world_y = y_in_view + view_y
                console.print(x_in_view + x, y_in_view + y, Terrain.to_symbol(self.terrain_at(world_x, world_y)))
        
        for to_render in self.objects:
            # Draw all the objects
            x_in_view = to_render.x - view_x
            y_in_view = to_render.y - view_y
            if x_in_view >= 0 and x_in_view < width and y_in_view >= 0 and y_in_view < height:
                console.print(x_in_view + x, y_in_view + y, to_render.symbol, fg=to_render.fg)
        
        
class PlayingState(GameState):
    """
    State for playing the game.
    
    Walk around as an @.
    """
    
    def __init__(self, world: GameWorld) -> None:
        """
        Set up a fresh game state.
        """
        
        # Keep the world
        self.world = world
        
        # Keep track of log messages and their counts
        self.logs: list[tuple[str, int]] = []
                
        self.log(f"You enter level {self.world.level_number} of the dungeon. You start getting {self.world.level_domain} vibes.")
        
    def log(self, message: str) -> None:
        """
        Store a message to the log of game event messages.
        """
        if len(self.logs) > 0 and self.logs[-1][0] == message:
            # A duplicate. Increase the count.
            self.logs[-1] = (self.logs[-1][0], self.logs[-1][1] + 1)
        else:
            self.logs.append((message, 1))
    
    def render_to(self, console: tcod.console.Console) -> None:       
        # Compute layout
        STATUS_HEIGHT = 2
        LOG_HEIGHT = 4
        
        # Clear screen
        console.clear()
        
        # Draw the status bar to report score and current item
        held_item = self.world.player.get_held_item()
        if held_item is not None:
            held_item_name = held_item.indefinite_name()
        else:
            held_item_name = "(nothing)"
        console.print_box(0, 0, console.width, STATUS_HEIGHT, f"Item (q/e): {held_item_name}", alignment=tcod.libtcodpy.LEFT)
        console.print_box(0, STATUS_HEIGHT - 1, console.width, STATUS_HEIGHT, f"{self.world.player.value} gp {self.world.player.health}/{self.world.player.max_health} HP", alignment=tcod.libtcodpy.LEFT)
        
        
        
        # Frame the world
        console.draw_frame(0, STATUS_HEIGHT, console.width, console.height - LOG_HEIGHT - STATUS_HEIGHT, GAME_NAME)
        
        # Draw a world view inset in the frame
        self.world.draw(console, 1, STATUS_HEIGHT + 1, console.width - 2, console.height - LOG_HEIGHT - STATUS_HEIGHT - 2)
 
        # Draw the log messages
        log_start_height = console.height - LOG_HEIGHT
        for log_message, log_count in reversed(self.logs):
            # Lay out log messages newest at the top, older below
            if log_count > 1:
                # Duplicate messages are expressed with counts
                log_message += f" x{log_count}"
            console.print(0, log_start_height, "•")
            log_start_height += console.print_box(1, log_start_height, console.width - 1, LOG_HEIGHT, log_message)
            if log_start_height >= console.height:
                break
            
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
        # WASD keys
        KeySym.a: (-1, 0),
        KeySym.d: (1, 0),
        KeySym.w: (0, -1),
        KeySym.s: (0, 1),
    }
    
    def handle_event(self, event: Optional[tcod.event.Event]) -> Optional[GameState]:
        if isinstance(event, tcod.event.KeyDown) and event.sym in self.DIRECTION_KEYS:
            # The player wants to move.
            direction = self.DIRECTION_KEYS[event.sym]
            
            next_x = self.world.player.x + direction[0]
            next_y = self.world.player.y + direction[1]
            
            obstruction = self.world.object_at(next_x, next_y)
            if obstruction is None:
                terrain = self.world.terrain_at(next_x, next_y)
                if terrain == Terrain.FLOOR:
                    # You can just move there
                    self.world.player.x = next_x
                    self.world.player.y = next_y
                else:
                    self.log("Impassable terrain!")
            elif isinstance(obstruction, Enemy):
                # Time to fight!
                
                weapon: Optional[Loot] = self.world.player.get_held_item()
                if weapon is not None:
                    damage_to_enemy, effectiveness = weapon.hit(obstruction)
                    damage_to_player, _ = obstruction.hit(weapon)
                else:
                    damage_to_enemy, effectiveness = self.world.player.hit(obstruction)
                    damage_to_player, _ = obstruction.hit(self.world.player)
                
                obstruction.health = max(0, obstruction.health - damage_to_enemy)
                self.world.player.health = max(0, self.world.player.health - damage_to_player)
                # Add a space around the effectiveness
                effectiveness = "  " + effectiveness + (" " if effectiveness != "" else "")
                if weapon is not None:
                    hit_message = f"You{effectiveness}attack {obstruction.definite_name()} with {weapon.definite_name()} for {damage_to_enemy} damage!"
                else:
                    hit_message = f"You{effectiveness}attack {obstruction.definite_name()} for {damage_to_enemy} damage!"
                if damage_to_player > 0:
                    hit_message += f" You take {damage_to_player}!"
                if obstruction.health > 0:
                    hit_message += f" Now {obstruction.nominative_pronoun} {obstruction.has_have} {obstruction.health}/{obstruction.max_health} HP."
                    self.log(hit_message)
                else:
                    # It is dead now
                    self.world.remove_object(obstruction)
                    hit_message += f" You kill {obstruction.accusative_pronoun}!"
                    self.log(hit_message)
                    
                    # Take its stuff
                    loot = random.choice(obstruction.inventory) if len(obstruction.inventory) > 0 else None
                    if loot is not None:
                        self.world.player.acquire_loot(loot)
                        self.log(f"You loot {loot.indefinite_name()}, worth {loot.value} gold!")
                        if len(self.world.player.inventory):
                            # Auto-equip the first item
                            self.world.player.next_item()
                            rarity_article = "a" if loot.rarity != "uncommon" else "an"
                            self.log(f"You ready {loot.definite_name()}, {rarity_article} {loot.rarity} item of the {loot.elemental_domain} domain.")
                
                # Check for winning
                if not self.world.has_enemies():
                    self.log("All enemies have been defeated!")
                    # Go to a victory state
                    return EndGameState(self.world, self.logs, self.world.level_number + 1, self.world.player, (0, 255, 0), f"Level Cleared!\nYou beat level {self.world.level_number}!", "Next Level")
                    
                # But if you don't win on the last attack you lose!
                if self.world.player.health <= 0:
                    self.log("You have died!")
                    return EndGameState(self.world, self.logs, 1, Player(), (255, 0, 0), f"You Lose!\nYou were defeated on level {self.world.level_number} by {obstruction.indefinite_name()}.")
                
            else:
                # The player is bumping something.
                self.log(f"Your path is obstructed by {obstruction.indefinite_name()}!")
        elif isinstance(event, tcod.event.KeyDown) and event.sym in (KeySym.q, KeySym.e):
            # Changing item
            old_item = self.world.player.get_held_item()
            if event.sym == KeySym.q:
                self.world.player.previous_item()
            elif event.sym == KeySym.e:
                self.world.player.next_item()
            new_item = self.world.player.get_held_item()
            if new_item != old_item:
                if new_item is not None:
                    rarity_article = "a" if new_item.rarity != "uncommon" else "an"
                    self.log(f"You ready {new_item.definite_name()}, {rarity_article} {new_item.rarity} item of the {new_item.elemental_domain} domain.")
                else:
                    self.log(f"You put away {old_item.definite_name()}.")
            
        # Don't change state by default
        return None
                
class EndGameState(PlayingState):
    """
    Acts like the normal in-game state, but you can't play and it shows a banner.
    """
    
    def __init__(self, world: GameWorld, logs: list[tuple[str, int]], next_level: int, next_player: Player, color: tuple[int, int, int], message: str, continue_message: str = "Play Again") -> None:
        super().__init__(world)
        # Keep the passed-in logs
        self.logs += logs
        # Remember the color to use
        self.color = color
        # And the message to display.
        self.message = message
        # And the question to ask
        self.continue_message = continue_message
        # Remember what level number to do next, if any
        self.next_level = next_level
        # And what player object to use
        self.next_player = next_player
        
    
    def render_to(self, console: tcod.console.Console) -> None:
        # First draw as normal
        super().render_to(console)
        
        # Then draw a big victory banner
        BANNER_WIDTH = 30
        BANNER_HEIGHT = 8
        banner_x = console.width // 2 - BANNER_WIDTH // 2
        banner_y = console.height // 2 - BANNER_HEIGHT // 2
        console.draw_frame(banner_x, banner_y, BANNER_WIDTH, BANNER_HEIGHT, decoration="╔═╗║ ║╚═╝", fg=self.color)
        console.print_box(banner_x + 1, banner_y + 1, BANNER_WIDTH - 2, BANNER_HEIGHT - 2 - 1, self.message, fg=self.color, alignment=tcod.libtcodpy.CENTER)
        console.print_box(banner_x + 1, banner_y + BANNER_HEIGHT - 2, BANNER_WIDTH - 2, 1, f"{self.continue_message} [Y/N]?", fg=self.color, alignment=tcod.libtcodpy.CENTER)
        
    def handle_event(self, event: Optional[tcod.event.Event]) -> Optional[GameState]:
        if isinstance(event, tcod.event.KeyDown):
            if event.sym in (tcod.event.KeySym.ESCAPE, tcod.event.KeySym.n):
                # Let the user quit at the end
                raise SystemExit()
            elif event.sym == tcod.event.KeySym.y:
                # Start a new level
                return LoadingState(self.next_level, self.next_player)
        
class LoadingState(GameState):
    """
    State for loading the game.
    """
    
    def __init__(self, level_number: int, player: Player):
        """
        Make a new loading state for the gievn level, using the given player.
        """
        self.world = GameWorld(level_number, player)
        self.process = self.world.generate_level(ProceduralGenerator())
        self.progress = (0, 0, "Loading")
    
    def render_to(self, console: tcod.console.Console) -> None:
        console.clear()
        console.draw_frame(0, 0, console.width, console.height, f"Loading Level {self.world.level_number}")
        
        bar_box_height = 3
        bar_box_width = console.width - 4
        console.draw_frame(2, console.height // 2 - bar_box_height // 2, bar_box_width, bar_box_height, f"{self.progress[0]:n}/{self.progress[1]:n}")
        bar_width = bar_box_width - 2
        bar_height = 1
        
        # Make a bar of semigraphic characters
        EMPTY_COLOR = (127, 0, 0)
        FULL_COLOR = (255, 0, 0)
        bar_graphic = tcod.image.Image(bar_width * 2, bar_height * 2)
        bar_graphic.clear(EMPTY_COLOR)
        bar_filled_px = round(bar_width * 2 * self.progress[0] / self.progress[1]) if self.progress[1] > 0 else 0
        for x in range(bar_filled_px):
            for y in range(bar_height * 2):
                bar_graphic.put_pixel(x, y, FULL_COLOR)
        console.draw_semigraphics(bar_graphic, 3, console.height // 2 - bar_height // 2)
        console.print_box(4, console.height // 2 + bar_box_height + 1, console.width - 8, console.height // 2 - bar_box_height - 1, f"{self.progress[2]}", alignment=tcod.libtcodpy.CENTER)
    
    def handle_event(self, event: Optional[tcod.event.Event]) -> Optional[GameState]:
        # Make progress
        try:
            self.progress = self.process.send(None)
        except StopIteration:
            # Now it is done
            return PlayingState(self.world)
            
            
    def get_wait(self) -> float:
        return 0
        

class ConsoleManager:
    """
    Class to let only one thread work on the tcod console at a time.
    """
    
    def __init__(self, width: int, height: int) -> None:
        """
        Make a new ConsoleManager to manage a console.
        """
        self.lock = Lock()
        self.console = tcod.console.Console(width, height)
    
    @contextmanager
    def with_console(self) -> Generator[tcod.console.Console, None, None]:
        """
        Get access to the console to draw on or to render.
        """
        with self.lock:
            yield self.console
    
    def resize(self, width: int, height: int) -> None:
        """
        Change the console size.
        
        Cannot be called inside with_console.
        """
        
        with self.lock:
            self.console = tcod.console.Console(width, height)        

def force_min_size(context: tcod.context.Context) -> None:
    """
    Force the window to be at least a minimum size.
    """
    MIN_WINDOW_WIDTH = 800
    MIN_WINDOW_HEIGHT = 600
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

    # We can't block the event wait loop or Windows will complain we're "not responding".
    # So we get events in one thread and queue them up, and handle them and also render in another thread.
    # The main thread just feeds the OS and the queue.
    event_queue = Queue()
    # And we want to send exceptions back
    error_queue = Queue()
    
    with tcod.context.new(tileset=tileset, title=GAME_NAME) as context:
        force_normal_shape(context)
        force_min_size(context)
        width, height = context.recommended_console_size()
        console_manager = ConsoleManager(width, height)
        
        def handle_queue():
            try:
                state = LoadingState(1, Player())
                
                while True:
                    # Main game loop
                    with console_manager.with_console() as console:
                        # Render the current game state, while we know the console isn't being put on the screen.
                        state.render_to(console)
                
                    try:
                        event = event_queue.get(timeout=state.get_wait())
                    except Empty:
                        event = None
                    
                    if isinstance(event, tcod.event.Quit) or isinstance(event, tcod.event.WindowEvent) and event.type == "WindowClose":
                        print("Stopping worker thread")
                        return
                    else:
                        next_state = state.handle_event(event)
                        if next_state is not None:
                            # Adopt the next state
                            state = next_state
            except BaseException as e:
                # Pass errors along, including SystemExit
                error_queue.put(e)
                return
                
        
        worker_thread = Thread(target=handle_queue, daemon=True)
        worker_thread.start()


        last_window_size = context.sdl_window.size
        while True: 
            # Event pumping loop
            
            with console_manager.with_console() as console:
                # Show console, while we know it isn't being rendered to
                context.present(console, keep_aspect=True)
            
            for event in tcod.event.wait(timeout=0.1):
                if isinstance(event, tcod.event.Quit) or isinstance(event, tcod.event.WindowEvent) and event.type == "WindowClose":
                    # Flush the existing queue so we can quit right away
                    try:
                        while True:
                            event_queue.get(block=False)
                    except Empty:
                        pass
                event_queue.put(event)
                if isinstance(event, tcod.event.Quit) or isinstance(event, tcod.event.WindowEvent) and event.type == "WindowClose":
                    while worker_thread.is_alive():
                        # Keep the window responsive while the worker terminates
                        with console_manager.with_console() as console:
                            console.clear()
                            console.print(0, 0, "Quitting...")
                            context.present(console, keep_aspect=True)
                        for event in tcod.event.wait(timeout=0.1):
                            pass
                    worker_thread.join()
                    sys.exit(0)
                elif isinstance(event, tcod.event.WindowResized):
                    # Manage window resizing and console resizing (while nobody is drawing)
                    if context.sdl_window.size != last_window_size:
                        force_min_size(context)
                        width, height = context.recommended_console_size()
                        console_manager.resize(width, height)
                        last_window_size = context.sdl_window.size
            try:
                # Re-raise any errors on the main thread.
                raise error_queue.get(block=False)
            except Empty:
                pass
                    
                    

if __name__ == "__main__":
    main()

