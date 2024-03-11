REM Create multi-file build in dist/<name>
pyinstaller -y --noconsole --onedir main.py -i icon.ico -n "False Ghost"

REM Now do all the other assets because I can't get e.g. --add-data "Curses_square_24.png;." to work.
REM insert llama-cpp-python DLLs that don't get picked up, from a user installation on Python 3.11
mkdir "dist\False Ghost\_internal\llama_cpp"
copy "%APPDATA%\Python\Python311\site-packages\llama_cpp\llama.dll" "dist\False Ghost\_internal\llama_cpp\llama.dll"
copy "%APPDATA%\Python\Python311\site-packages\llama_cpp\llava.dll" "dist\False Ghost\_internal\llama_cpp\llava.dll"
REM Copy assets
copy Curses_square_24.png "dist\False Ghost\Curses_square_24.png"
mkdir "dist\False Ghost\grammars"
copy grammars\*.gbnf "dist\False Ghost\grammars\"

REM Copy docs
copy README.md "dist\False Ghost\"

pause