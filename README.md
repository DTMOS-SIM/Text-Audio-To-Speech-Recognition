# Text-Audio-To-Speech-Recognition
Speech recognition comparison using Librosa and Vosk library

### Theorical Design
DeepSpeech and Vosk has long been a point of comparison between their text-to-speech accuracy. This repository has been purposefully made to ensue individual's accuracy based on a series of speeches recorded in multiple languages and tested on both system. Outcome is surprisingly significant, have a shot at this project for those who are into speech-recognition technology.

### Futher Improvements
- [ ] Class-based design implementation
    - [ ] Object Initialisation()
        - [ ] Vosk Class
        - [ ] DeepSpeech Class
- [ ] Callable API endpoint for further simplification of running
    - [ ] Initialisation of class with parmaters 
        - [ ] :file_folder: Codec
        - [ ] :black_square_button: Frames
    - [ ] :file_folder: LoadFile()
    - [ ] :sound: Noise_Reduction() - cuts background noise for improved speech accuracy
    - [ ] :arrows_counterclockwise: Word_Error_Rate() - calculation of word error count
    - [ ] :hourglass: Train() - computes the result for each audio into string format
- [ ] Provide User documentation:
    - [ ] Installation Process (Dependencies and Versions)
    - [ ] Classes and Objects
    - [ ] Step-by-Step examples
