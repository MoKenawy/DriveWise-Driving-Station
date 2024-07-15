from collections.abc import Callable
from os import error
from threading import Thread
from typing import Any, Iterable, Mapping
from pynput import keyboard
from ipaddress import IPv4Address
import requests



class CarController():
    def __init__(self, ip, port):
        self.ip = IPv4Address(ip)
        self.port = port
        print(f"CarControllerListener is Running for \nip: {self.ip}\nport : {self.port}")

    def move(self,state):
        http_url = "http://"+ str(self.ip) + ":"+ str(self.port) + "/" +state
        print(http_url)
        try:
            r = requests.get(url = http_url)
            return r.json
        except requests.exceptions.ConnectionError:
            print("Car Controller : Connection Error [Request raised a network problem (e.g. DNS failure, refused connection, etc)]")
        except requests.exceptions.HTTPError:
            print("Car Controller : Http Error [invalid HTTP response]")
        except requests.exceptions.Timeout:
            print("Car Controller : Request Timeout Error")
        except requests.exceptions.TooManyRedirects:
            print("Car Controller : Request TooManyRedirects Error")
        except requests.exceptions.RequestException :
            print("Car Controller : Something went wrong while sending the request")
        except Exception as e:
            print(e)
            

class CarControllerListener(Thread):
    def __init__(self, ip, port,group: None = None, target: Callable[..., object] | None = None, name: str | None = None, args: Iterable[Any] = ..., kwargs: Mapping[str, Any] | None = None, *, daemon: bool | None = None) -> None:

        super().__init__(group, target, name, args, kwargs, daemon=daemon)
        self.carController = CarController(ip, port)

    def run(self) -> None:
        super().run()

        # Collect events until released
        with keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release) as listener:
            listener.join()

        # ...or, in a non-blocking fashion:
        listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        listener.start()



    def on_press(self, key):
        ...

    def on_release(self,key):

        self.controlCar(key)

        if key == keyboard.Key.esc:
            # Stop listener
            return False
        
    def controlCar(self, key):
        if key == keyboard.Key.up:
            self.carController.move("forward")
        elif key == keyboard.Key.left: 
            self.carController.move("left")
        elif key == keyboard.Key.down: 
            self.carController.move("backward")
        elif key == keyboard.Key.right: 
            self.carController.move("right")



        


if __name__ == '__main__':
    carController = CarControllerListener("127.0.0.1",8000)
    carController.start()
