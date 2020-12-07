import arcade
import math
from pymunk.vec2d import Vec2d
import time

from dataclasses import dataclass
from typing import List

SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 800
SCREEN_TITLE = "Under Development"
SCALING = 0.1
PLAYER_1_SPEED = 5
PLAYER_2_SPEED = 3

TILE_SCALING = 1.5

# Keybindings

MOVE_MAP_PLAYER_1 = {
    arcade.key.W: Vec2d(0, 1),
    arcade.key.S: Vec2d(0, -1),
    arcade.key.A: Vec2d(-1, 0),
    arcade.key.D: Vec2d(1, 0),
}

MOVE_MAP_PLAYER_2 = {
    arcade.key.I: Vec2d(0, 1),
    arcade.key.K: Vec2d(0, -1),
    arcade.key.J: Vec2d(-1, 0),
    arcade.key.L: Vec2d(1, 0),
}

# Classes


class DashState():
    def process_input(self, key):
        return


class DefaultState():

    def __init__(self):
        self.time_since_dmg=1000

    def update(self,delta_time):
        self.time_since_dmg+=delta_time

    def take_damage(self,player,damage):
        if self.time_since_dmg>1:
            self.time_since_dmg=0
            player.health -= damage
            if player.health <= 0:
                player.health = 0
                print("I'm legally dead")

            num_dashes = int(player.health/10)
            text = f"|"+'_'*num_dashes+' '*(10-num_dashes)+'|'
            print(text)

    def on_key_press(self, player, key):
        if key in player.MOVE_MAP:
            player.keys_pressed[key] = True
            player.move_direction = sum(
                player.keys_pressed[k] * player.MOVE_MAP[k] for k in player.keys_pressed).normalized()
            player.change_y = player.move_direction.y * player.speed
            player.change_x = player.move_direction.x * player.speed

            if player.move_direction != Vec2d(0, 0):
                player.facing_direction = player.move_direction

        elif key == arcade.key.LSHIFT and time.time()-player.time_dashed > 2:
            print("dash")
            player.speed *= 2
            player.time_dashed = time.time()

    def on_key_release(self, player, key):
        if key in player.MOVE_MAP:
            player.keys_pressed[key] = False
            player.move_direction = sum(
                player.keys_pressed[k] * player.MOVE_MAP[k] for k in player.keys_pressed).normalized()
            player.change_y = player.move_direction.y * player.speed
            player.change_x = player.move_direction.x * player.speed

            if player.move_direction != Vec2d(0, 0):
                player.facing_direction = player.move_direction


class Bullet(arcade.Sprite):
    def __init__(self, filename, scaling, max_bounces, speed=5):
        super().__init__(filename, scaling)
        self.bounces = 0
        self.max_bounces = max_bounces
        self.speed = speed
        self.color = arcade.color.BRIGHT_GREEN

    def update(self):
        if self.bounces > self.max_bounces:
            self.remove_from_sprite_lists()


class Player(arcade.Sprite):
    def __init__(self, filename, scaling, MOVE_MAP, health=100, speed=5):
        super().__init__(filename, scaling)
        self.speed = speed
        self.health = health
        self.move_direction = Vec2d(0, 0)
        self.facing_direction = Vec2d(1, 0)
        self.MOVE_MAP = MOVE_MAP
        self.keys_pressed = {k: False for k in self.MOVE_MAP}

        self.state = DefaultState()

    def update(self,delta_time):
        self.state.update(delta_time)

    def shoot(self):
        bullet = Bullet("./sprites/weapon_gun.png", 1, 3)
        bullet.change_x = self.facing_direction.x * bullet.speed
        bullet.change_y = self.facing_direction.y * bullet.speed

        start_x = self.center_x
        start_y = self.center_y
        bullet.center_x = start_x
        bullet.center_y = start_y
        angle = math.atan2(self.facing_direction.y, self.facing_direction.x)

        bullet.angle = math.degrees(angle)

        return bullet

    def take_damage(self, damage):
        self.state.take_damage(self,damage)

    def on_key_press(self, key, modifiers):
        self.state.on_key_press(player=self, key=key)

    def on_key_release(self, key, modifiers):
        self.state.on_key_release(player=self, key=key)


class Shooter(arcade.Window):
    """Main welcome window
    """

    def __init__(self):
        """Initialize the window
        """
        # Call the parent class constructor
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.bullets = arcade.SpriteList()
        self.wall_list = arcade.SpriteList()
        self.deadly_list = arcade.SpriteList()
        self.players = arcade.SpriteList()
        self.all_sprites = arcade.SpriteList()

        # Physics engine currently only handles player-wall collisions
        self.physics_engine = None

    def setup(self):

        # --- Load in a map from the tiled editor ---

        # Name of map file to load
        map_name = "./maps/map_1/map.tmx"
        # Name of the layer in the file that has our platforms/walls
        walls_layer_name = 'walls'
        # Name of the layer that has floor
        floor_layer_name = 'floor'
        # Name of the layer that has deadly tiles
        deadly_layer_name = "toxic"

        my_map = arcade.tilemap.read_tmx(map_name)
        self.wall_list = arcade.tilemap.process_layer(map_object=my_map,
                                                      layer_name=walls_layer_name,
                                                      scaling=TILE_SCALING,
                                                      use_spatial_hash=True)

        #-- Floor
        self.floor_list = arcade.tilemap.process_layer(
            my_map, floor_layer_name, TILE_SCALING)
        #-- Deadly
        self.deadly_list = arcade.tilemap.process_layer(
            my_map, deadly_layer_name, TILE_SCALING)

        self.all_sprites.extend(self.deadly_list)
        self.all_sprites.extend(self.floor_list)  # extend appends spriteList
        self.all_sprites.extend(self.wall_list)

        bullet = Bullet("./sprites/weapon_gun.png", SCALING, 3)
        bullet.center_y = self.height / 2
        bullet.left = 200
        bullet.change_x = 3
        bullet.change_y = 3

        self.bullets.append(bullet)
        self.all_sprites.append(bullet)

        # Player setups
        self.player1 = Player("sprites/duck_small.png", 0.2, MOVE_MAP_PLAYER_1)
        self.player1.center_y = self.height / 2
        self.player1.left = 100

        self.players.append(self.player1)

        # Player - wall Collisions
        self.physics_engine = arcade.PhysicsEngineSimple(
            self.player1, self.wall_list)

        self.time_since_dmg = 10000
        # self.all_sprites.append(player1)

    def on_draw(self):
        """Called whenever you need to draw your window
        """

        # Clear the screen and start drawing
        arcade.start_render()
        self.all_sprites.draw()
        self.bullets.draw()

        self.players.draw()

    def on_update(self, delta_time):

        # Bullet bounces
        for bullet in self.bullets:
            bounced = False
            bullet.center_x += bullet.change_x

            walls_hit = arcade.check_for_collision_with_list(
                bullet, self.wall_list)

            for wall in walls_hit:
                if bullet.change_x > 0:
                    bullet.right = wall.left
                elif bullet.change_x < 0:
                    bullet.left = wall.right

            if len(walls_hit) > 0:
                bullet.change_x *= -1
                bullet.bounces += 1
                bounced = True

            bullet.center_y += bullet.change_y

            walls_hit = arcade.check_for_collision_with_list(
                bullet, self.wall_list)

            for wall in walls_hit:
                if bullet.change_y > 0:
                    bullet.top = wall.bottom
                elif bullet.change_y < 0:
                    bullet.bottom = wall.top

            if len(walls_hit) > 0:
                bullet.change_y *= -1
                bullet.bounces += 1
                bounced = True

            if bounced == True:
                angle = math.atan2(bullet.change_y, bullet.change_x)
                bullet.angle = math.degrees(angle)

        if self.player1.collides_with_list(self.deadly_list):
            self.player1.color = arcade.color.AFRICAN_VIOLET
            self.player1.take_damage(10)
        else:
            self.player1.color = arcade.color.NON_PHOTO_BLUE

        for player in self.players:
            player.change_x = player.move_direction.x*player.speed
            player.change_y = player.move_direction.y*player.speed

        self.player1.update(delta_time)

        self.physics_engine.update()

        self.all_sprites.update()

    def on_key_press(self, key, modifiers):

        if key == arcade.key.C:
            bullet = self.player1.shoot()
            self.bullets.append(bullet)
            self.all_sprites.append(bullet)

        elif key == arcade.key.N:
            bullet = self.player2.shoot()
            self.bullets.append(bullet)
            self.all_sprites.append(bullet)

        for player in self.players:
            player.on_key_press(key, modifiers)

    def on_key_release(self, key, modifiers):

        for player in self.players:
            player.on_key_release(key, modifiers)


# Main code entry point
if __name__ == "__main__":
    app = Shooter()
    app.setup()
    arcade.run()
