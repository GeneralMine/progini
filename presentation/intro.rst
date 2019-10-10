:css: style.css

.. title:: Hacktoberfest 2019 - Project Lantern

----

:data-x: 0
:data-y: 0
:data-scale: 1

.. image:: images/hacktoberfest.png

Hacktoberfest 2019
==================

Project Lantern
---------------

Markus Pöschl & Michael Ziegler

----

:data-x: r2400
:data-y: 0

.. image:: images/PG_Logo_Web.png
   :height: 260px

Sponsor
-------

PIXEL Group GmbH

Simon Ashdown


----

.. image:: images/menu.png

Important things first
----------------------

* Schnitzel will be ready right after the presentation.
* Drinks and snacks are in the kitchen.

We also have vegetarians covered. ;)


----

.. image:: images/hacktoberfest-official.svg
   :width: 700px

Hacktoberfest
-------------

* month-long celebration of open source software
	* support for open-source
* sponsored by DigitalOcean and DEV


----

:data-x: r0
:data-y: r1000

.. image:: images/hacktoberfest-official.svg
   :width: 700px

.. image:: images/progress.png
   :width: 500px

Rules
-----

* Register at https://hacktoberfest.digitalocean.com/
* Open 4 Pull Requests
	* Any repository on GitHub
	* Between 1 - 31 October
* (Wait 7 days)
* Profit (T-Shirt, Stickers and more)

Your Progress: https://hacktoberfest.digitalocean.com/profile

----

:data-x: r2400
:data-y: 0

.. image:: images/lantern.jpg

Project Lantern
---------------

Codename: progini


----

:id: algorithm-id
:data-x: r0
:data-y: r1000

.. image:: images/finger-track.jpg

Finger-Tracking
---------------

* Python
* OpenCV


----

.. image:: images/images.png

How does it work
----------------

1. Take a grayscale image
2. Apply erosion
3. Detect motion by comparing with previous image
4. On motion search for the biggest contour
5. Take the point with the biggest distance to the sides
6. Move the mouse pointer there


----

:data-x: r2400
:data-y: 0

.. image:: images/blox.png

Lantern-Blox
------------

* Maulwurf-like demo game
* Click the boxes when they appear
* Javascript + executed in full screen browser on lantern projector


----

.. image:: images/develop.png

Now is your turn
----------------

* Develop your own app / game
	* UI for home automation
	* Memory
	* ...
* Improve the finger-detection
	* Alternative Mouse-Handling
	* ...

Source code
-----------

https://github.com/hacktoberfestmunich/progini

* Open a PR for your work


----

.. image:: images/lanter-live.jpg
   :width: 300px
.. image:: images/pi.png
   :width: 300px


Testing ?
---------

* Lantern to test
	* Push as PR and notify us

* Raspberry Pi Simulation
	* No finger detection
	* Mouse simulates the finger input
	* More for performance testing

----

Hacking the image processing
----------------------------

* The finger tracking software does not suite your needs?
	* No problem, hack it!

* Suggested Workflow
	* Develope the ui / application you want.
	* Try it on the lantern.
	* Does it work?
		* Fine!
	* If not:
		* Enable the debug mode and capture a series of input images from the camera
		* Now you can feed back the captured images to the finger control software
* Hint: Read -h of the finger control program

.. image:: images/help_finger_control.png
