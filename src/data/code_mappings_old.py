# This list was extracted from the raw csv. We ignore most at-bat events, however we are still interested
# in differentiating between singles, doubles, triples, and home runs. In this case, PitchResult is used "incorrectly"
from src.model.at_bat import PitchResult
from src.model.pitch_type import PitchType

old_at_bat_event_mapping = {
    'Double': PitchResult.HIT_DOUBLE,
    'Single': PitchResult.HIT_SINGLE,
    'Triple': PitchResult.HIT_TRIPLE,
    'Home Run': PitchResult.HIT_HOME_RUN,
    # Groundout
    # Strikeout
    # Strikeout
    'Walk': PitchResult.HIT_SINGLE,
    # Runner Out
    # Flyout
    # Forceout
    # Pop Out
    'Intent Walk': PitchResult.HIT_SINGLE,
    # Lineout
    'Hit By Pitch': PitchResult.HIT_SINGLE,
    # Grounded Into DP
    # Sac Bunt
    # Fielders Choice
    # Bunt Groundout
    'Field Error': PitchResult.HIT_SINGLE,
    # Double Play
    # Sac Fly
    # Fielders Choice Out
    # Bunt Pop Out
    'Catcher Interference': PitchResult.HIT_SINGLE,
    # Strikeout - DP
    # Batter Interference
    # Sac Fly DP
    # Bunt Lineout
    # Sacrifice Bunt DP
    # Triple Play
}

old_pitch_type_mapping = {
    'CH': PitchType.CHANGEUP,
    'CU': PitchType.CURVE,
    'EP': PitchType.CURVE,  # Eephus is a type of slow curveball
    'FC': PitchType.CUTTER,
    'FF': PitchType.FOUR_SEAM,
    'FO': None,  # Pitchout doesn't fit in the given categories
    'FS': PitchType.CHANGEUP,  # Splitter is often considered a type of changeup
    'FT': PitchType.TWO_SEAM,
    'IN': None,  # Intentional ball doesn't fit in the given categories
    'KC': PitchType.CURVE,
    'KN': PitchType.CURVE,  # Knuckleball is distinct but closer to a curve in action
    'PO': None,  # Pitchout doesn't fit in the given categories
    'SC': PitchType.CHANGEUP,  # Screwball acts like a changeup with reverse break
    'SI': PitchType.TWO_SEAM,
    'SL': PitchType.SLIDER,
    'UN': None,  # Unknown types cannot be classified
}

old_pitch_result_mapping = {
    'B': PitchResult.CALLED_BALL,  # Ball
    '*B': PitchResult.CALLED_BALL,  # Ball in dirt
    'I': PitchResult.CALLED_BALL,  # Intentional ball
    'P': PitchResult.CALLED_BALL,  # Pitchout
    'C': PitchResult.CALLED_STRIKE,  # Called strike
    'S': PitchResult.SWINGING_STRIKE,  # Swinging strike
    'W': PitchResult.SWINGING_STRIKE,  # Swinging strike (blocked)
    'Q': PitchResult.SWINGING_STRIKE,  # Swinging pitchout
    'T': PitchResult.SWINGING_STRIKE,  # Foul tip
    'L': PitchResult.SWINGING_STRIKE,  # Foul bunt
    'F': PitchResult.SWINGING_FOUL,  # Regular foul
    'R': PitchResult.SWINGING_FOUL,  # Foul pitchout
    'D': PitchResult.HIT_SINGLE,  # Hit
    'E': PitchResult.HIT_SINGLE,  # Hit, with runs scored
    'H': PitchResult.HIT_SINGLE,  # Hit by pitch, perhaps should be a separate category
    'X': PitchResult.HIT_OUT,  # In play, out(s)
}