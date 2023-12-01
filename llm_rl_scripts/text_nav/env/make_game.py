import random
from typing import Tuple
import numpy as np
import networkx as nx
import textworld
from textworld import GameMaker, GameOptions
from textworld.generator.game import Event, Quest, Game
from textworld.generator.game import GameOptions
from textworld.logic import Proposition


def compile_game(game, options: GameOptions, hide_location: bool = True) -> str:
    grammar_flags = {
        "theme": "house",
        "include_adj": False,
        "only_last_action": True,
        "blend_instructions": True,
        "blend_descriptions": True,
        "hide_location": hide_location,
        "instruction_extension": []
    }
    rng_grammar = np.random.RandomState(1234)
    grammar = textworld.generator.make_grammar(grammar_flags, rng=rng_grammar)
    game.change_grammar(grammar)

    game_file = textworld.generator.compile_game(game, options)
    return game_file


def build_game(options: GameOptions) -> Game:
    M = GameMaker(options)

    bedroom = M.new_room("bedroom")
    office = M.new_room("office")
    bathroom = M.new_room("bathroom")
    livingroom = M.new_room("living room")
    kitchen = M.new_room("kitchen")
    diningroom = M.new_room("dining room")
    garden = M.new_room("garden")
    backyard = M.new_room("backyard")

    hallway1 = M.connect(bedroom.west, livingroom.east)
    hallway2 = M.connect(livingroom.west, diningroom.east)
    hallway3 = M.connect(diningroom.west, kitchen.east)
    hallway4 = M.connect(kitchen.north, garden.south)
    hallway5 = M.connect(livingroom.south, office.north)
    hallway6 = M.connect(office.east, bathroom.west)
    hallway7 = M.connect(bedroom.south, bathroom.north)
    hallway8 = M.connect(kitchen.west, backyard.east)


    table1 = M.new(type='s', name='table') 
    table2 = M.new(type='s', name='table')
    table3 = M.new(type='s', name='table')
    chest1 = M.new(type='c', name='chest')
    fridge = M.new(type='c', name='fridge')
    M.add_fact("closed", chest1)
    M.add_fact("closed", fridge)
    
    bedroom.add(table1)
    office.add(chest1)
    livingroom.add(table2)
    diningroom.add(table3)
    kitchen.add(fridge)

    food = M.new(type="f", name="stale food")
    distractor1 = M.new(type="f", name="fresh food")
    distractor2 = M.new(type="o", name="bowl")
    distractor3 = M.new(type="o", name="coffee cup")
    distractor4 = M.new(type="o", name="plate")
    distractor5 = M.new(type="o", name="utensils")
    distractor6 = M.new(type="f", name="fruit")
    
    table2.add(food)
    table2.add(distractor1)
    table1.add(distractor2)
    table3.add(distractor3)
    table3.add(distractor4)
    chest1.add(distractor5)
    fridge.add(distractor6)

    random_room = random.choice([bedroom, bathroom, office, livingroom, kitchen,
                                 diningroom, garden, backyard])

    M.player = M.new(type='P')
    M.set_player(random_room)
    
    quests = []

    food_consumed = Event(conditions={M.new_fact("eaten", food)})
    quests.append(Quest(win_events=[], fail_events=[food_consumed]))

    fridge_closed_with_food = Event(
        conditions={
            M.new_fact("in", food, fridge),
            M.new_fact("closed", fridge)
        })
    quests.append(Quest(win_events=[fridge_closed_with_food], reward=1))
    M.quests = quests

    G = compute_graph(M)  # Needed by the move(...) function called below.

    # Build walkthrough.
    current_room = random_room
    walkthrough = []

    # 0. Find the kitchen and read the cookbook.
    walkthrough += move(M, G, current_room, livingroom)
    current_room = livingroom

    walkthrough.append("take stale food from table")

    walkthrough += move(M, G, current_room, kitchen)
    
    walkthrough.append("open fridge")
    walkthrough.append("insert stale food into fridge")
    walkthrough.append("close fridge")

    M.set_walkthrough(walkthrough)
    
    metadata = {
        "walkthrough": walkthrough,
    }
    return M.build(), metadata


def build_and_compile_game(hide_location: bool = True) -> Tuple[Game, str]:
    options = GameOptions()
    options.grammar.include_adj = False
    options.grammar.only_last_action = True
    options.grammar.blend_instructions = True
    options.grammar.blend_descriptions = True
    options.grammar.hide_location = hide_location

    game, metadata = build_game(options)
    objective = ("Your objective is to put the stale food on the table in the living room"
                 " into the fridge in the kitchen.")
    game.objective = objective

    metadata["partial_information"] = hide_location
    game.metadata = metadata

    game_file = compile_game(game, options, hide_location)
    return game, game_file


## HELPER FUNCTIONS ##

def move(M, G, start, end):
    path = nx.algorithms.shortest_path(G, start.id, end.id)
    commands = []
    current_room = start
    for node in path[1:]:
        previous_room = current_room
        direction, current_room = [(exit.direction, exit.dest.src) for exit in previous_room.exits.values()
                                   if exit.dest and exit.dest.src.id == node][0]
        commands.append("go {}".format(direction))

    return commands


def compute_graph(M):
    G = nx.Graph()
    constraints = []
    G.add_nodes_from(room.id for room in M.rooms)

    def is_positioning_fact(proposition: Proposition):
        return proposition.name in ["north_of", "south_of", "east_of", "west_of"]

    positioning_facts = [fact for fact in M.facts if is_positioning_fact(fact)]
    for fact in positioning_facts:
        G.add_edge(fact.arguments[0].name, fact.arguments[1].name)
        constraints.append((fact.arguments[0].name, fact.name[:-3], fact.arguments[1].name))

    return G
