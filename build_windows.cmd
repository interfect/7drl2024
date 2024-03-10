REM Create multi-file build in dist/main
pyinstaller -y --noconsole --onedir main.py -i icon.ico

REM Now do all the other assets because I can't get e.g. --add-data "Curses_square_24.png;." to work.
REM insert llama-cpp-python DLLs that don't get picked up, from a user installation on Python 3.11
mkdir dist\main\_internal\llama_cpp
copy "%APPDATA%\Python\Python311\site-packages\llama_cpp\llama.dll" dist\main\_internal\llama_cpp\llama.dll
copy "%APPDATA%\Python\Python311\site-packages\llama_cpp\llava.dll" dist\main\_internal\llama_cpp\llava.dll
REM Copy assets
copy Curses_square_24.png dist\main\Curses_square_24.png
mkdir dist\main\grammars
copy grammars\*.gbnf dist\main\grammars\

pause