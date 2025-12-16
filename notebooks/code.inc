# -*- mode: org; -*-

#+NAME: hello
#+BEGIN_SRC python :exports none :results output
print('Welcome to CSUS!', end='')
#+END_SRC

#+NAME: first_word
#+BEGIN_SRC python :exports none :var line="foo bar" :results output
print(line.split(' ', 1)[0], end='')
#+END_SRC


#+NAME: strlen
#+BEGIN_SRC python :var line="18 characters long" :exports none :results output
print(len(line), end='')
#+END_SRC

#+NAME: short_hash
#+BEGIN_SRC python :var line="somehash" :exports none :results output
print(line[:4], end='')
#+END_SRC

#+NAME: hash_to_dir
#+BEGIN_SRC python :var line="somehash\n\n" :exports none :results output
print(f".git/objects/{line[:2]}/{line[2:]}".rstrip(), end='')
#+END_SRC


#+NAME: init
#+BEGIN_SRC shell :exports none :results output
rm -rf .git
rm -f *.{html,org,md,txt}
#+END_SRC
