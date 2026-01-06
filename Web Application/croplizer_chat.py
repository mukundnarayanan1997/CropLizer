# -*- coding: utf-8 -*-
"""
Croplizer Chat Logic
Handles the conversation flow, state management, and bridging to the main prediction engine.
"""

import re
import numpy as np
from datetime import datetime
import gradio as gr
import traceback
from textwrap import dedent

# --- Constants & Configuration ---
# Defaults for economic values if not collected in chat
ECONOMIC_DEFAULTS = {
    'FGPRICE_quintal': 3368.0, 'Urea_price': 250.0, 'DAP_price': 1400.0, 'MOP_price': 250.0,
    'Seed_price': 105.08, 'Labour_cost': 48533.0, 'Insecticides_cost': 2286.91,
    'Irrigation_cost': 2771.01, 'Insurance_cost': 64.61, 'Misc_cost': 314.44,
    'Interest_working_cost': 1362.0, 'Rent_owned_cost': 19761.0, 'Rent_leased_cost': 918.9,
    'Land_revenue_cost': 60.29, 'Depreciation_cost': 550.37, 'Interest_fixed_cost': 2610.97
}

# Mapping for readable labels back to model values
INTERPRETABLE_MAPS = {
    'EDU': {'No Formal Schooling': '1.0', 'Primary': '2.0', 'Matriculation': '3.0', 'Senior Secondary': '4.0', 'Bachelors': '5.0', 'Masters': '6.0', 'PhD': '7.0'},
    'SOPER': {'Low': '1.0', 'Medium': '2.0', 'High': '3.0'},
    'DCLASS': {'Very Lowland': '1.0', 'Lowland': '2.0', 'Mediumland': '3.0', 'Upland': '4.0'},
    'WSEV': {'None': '0.0', 'Low': '1.0', 'Medium': '2.0', 'High': '3.0'},
    'INSEV': {'None': '0.0', 'Low': '1.0', 'Medium': '2.0', 'High': '3.0'},
    'DISEV': {'None': '0.0', 'Low': '1.0', 'Medium': '2.0', 'High': '3.0'},
    'DRSEV': {'None': '0.0', 'Low': '1.0', 'Medium': '2.0', 'High': '3.0'},
    'FLSEV': {'None': '0.0', 'Low': '1.0', 'Medium': '2.0', 'High': '3.0'},
    'SEASON': {'Kharif': 'Kharif', 'Rabi': 'Rabi'},
    'GEN': {'Male': 'Male', 'Female': 'Female'},
    'SOCCAT': {'SC': 'SC', 'ST': 'ST', 'OBC': 'OBC', 'General': 'General'},
    'VARTYPE': {'Basmati': 'basmati', 'Local': 'local', 'Improved': 'improved', 'Hybrid': 'hybrid', 'Traditional Local': 'Traditional_Local'},
    'EST_line': {'Line': 'Line', 'Random': 'Random'},
    'EST_binary': {'Direct Seed': 'Direct_seed', 'Transplanted': 'Transplanted'},
    'IRRIAVA': {'Yes': 'yes', 'No': 'no'}
}

# --- Helper Functions ---

def date_str_to_doy(date_val):
    try:
        # Try parsing standard formats
        for fmt in ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"]:
            try:
                dt = datetime.strptime(str(date_val).strip(), fmt)
                return float(dt.timetuple().tm_yday)
            except ValueError:
                continue
        # Check if it's already a number (DOY)
        val = float(date_val)
        if 1 <= val <= 366:
            return val
        return np.nan
    except:
        return np.nan

def generate_questions():
    """Generates the sequential list of questions for the chatbot."""
    q = []

    # 1. Location
    q.append({
        'key': 'LOC', 
        'q': "👋 Hello! I'm your Farm Advisor. I can help optimize your yield.\n\nFirst, please tell me your **Latitude and Longitude** (e.g., '22.5, 88.3') or select 'Auto' to detect it.", 
        'type': 'loc', 
        'options': ['Auto']
    })

    # 2. Farm Details
    q.append({'key': 'CRLPARHA', 'q': "What is your current **Plot Area** in hectares? (e.g., 0.5)", 'type': 'float'})
    q.append({'key': 'SEASON', 'q': "Which **Season** are you planting in?", 'type': 'choice', 'options': list(INTERPRETABLE_MAPS['SEASON'].keys())})

    # 3. Soil & Water (Simplified for Chat)
    q.append({'key': 'SOPER', 'q': "How would you rate your **Soil Quality**?", 'type': 'choice', 'options': list(INTERPRETABLE_MAPS['SOPER'].keys())})
    q.append({'key': 'IRRIAVA', 'q': "Is **Irrigation Available**?", 'type': 'choice', 'options': list(INTERPRETABLE_MAPS['IRRIAVA'].keys())})

    # 4. Practices
    q.append({'key': 'VARTYPE', 'q': "What **Rice Variety** are you using?", 'type': 'choice', 'options': list(INTERPRETABLE_MAPS['VARTYPE'].keys())})
    q.append({'key': 'TDATE_yday', 'q': "What is your **Planting Date**? (Format: YYYY-MM-DD)", 'type': 'date'})
    q.append({'key': 'SRATE_kg', 'q': "Total **Seed Used** (in kg) for this entire plot?", 'type': 'float'})

    return q

CHAT_QUESTIONS = generate_questions()

# --- Main Logic ---

def process_turn(msg, history, state, lang_code, translator_fn, fetch_data_fn, predict_fn):
    """
    Handles a single turn of the conversation.
    Args:
        msg: User input text.
        history: Chat history list.
        state: Dictionary containing 'step' and 'data'.
        lang_code: Language code from UI.
        translator_fn: Function to translate text.
        fetch_data_fn: Function to get climate/soil data.
        predict_fn: Function to run the main prediction.
    """
    try:
        if state is None: state = {'step': 0, 'data': {}}
        if history is None: history = []

        msg = str(msg).strip()
        
        # Reset Logic
        if msg.lower() in ['reset', 'restart', 'start over', 'clear']:
            state = {'step': 0, 'data': {}}
            msg = "" # Treat as fresh start
            history = [] 
        
        step_idx = state.get('step', 0)
        
        # --- 0. STARTUP CHECK ---
        # Detect greeting or initial interaction. 
        # Check against translated 'Auto' if lang_code is not English.
        is_greeting = False
        if step_idx == 0 and not state.get('data'):
            if not msg:
                is_greeting = True
            else:
                # Check known greetings
                greetings = ['hi', 'hello', 'start', 'hey']
                if msg.lower() in greetings:
                    is_greeting = True
                
                # Check if msg is 'Auto' (English) or Translated 'Auto'
                # If it IS 'Auto', it's NOT a greeting, it's an answer to Q1.
                # So we only flag as greeting if it is NOT 'Auto'
                is_auto = msg.lower() == 'auto'
                if not is_auto and lang_code != 'English':
                     # Check translated auto
                     try:
                         auto_trans = translator_fn('Auto', lang_code)
                         if msg.lower() == auto_trans.lower():
                             is_auto = True
                     except: pass
                
                if is_auto:
                    is_greeting = False
                elif msg.lower() in greetings:
                    is_greeting = True

        if is_greeting:
            # Just display the first question
            first_q = CHAT_QUESTIONS[0]
            q_text = translator_fn(first_q['q'], lang_code)
            
            # Avoid duplicate greeting if it's already the last message
            if not history or history[-1][1] != q_text:
                history.append((None, q_text))
            
            # Translate Initial Options (e.g., 'Auto') using Tuple format (Label, Value)
            eng_options = first_q.get('options', [])
            trans_options = [(translator_fn(opt, lang_code), opt) for opt in eng_options] if eng_options else []
            
            return history, state, gr.update(visible=True, choices=trans_options, value=None)

        # --- 1. VALIDATE CURRENT INPUT ---
        if step_idx < len(CHAT_QUESTIONS):
            q_meta = CHAT_QUESTIONS[step_idx]
            key = q_meta['key']
            valid = False
            
            # Helper: Try to translate input back to English if it might be a translated option
            msg_english = msg
            if lang_code != 'English':
                try:
                    # Attempt to reverse translate user input to English for matching keys
                    # We assume translator_fn(text, 'English') will return English text
                     pass 
                except:
                    pass

            # A. Location Handling
            if key == 'LOC':
                matches = re.findall(r"[-+]?\d*\.\d+|\d+", msg)
                
                if len(matches) >= 2:
                    try:
                        lat, lon = float(matches[0]), float(matches[1])
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            state['data']['LAT'] = lat
                            state['data']['LONG'] = lon
                            valid = True
                            
                            fetch_msg = translator_fn("🌍 Coordinates received. Fetching soil & weather data...", lang_code)
                            history.append((msg, fetch_msg))
                            
                            try:
                                clim_soil = fetch_data_fn(lat, lon)
                                keys = ['total_precip','num_dry_days','avg_dspell_length','monsoon_onset','monsoon_length',
                                        'SoilGrids_bdod','SoilGrids_clay','SoilGrids_nitrogen','SoilGrids_ocd',
                                        'SoilGrids_phh2o','SoilGrids_sand','SoilGrids_silt','SoilGrids_soc']
                                
                                if clim_soil and len(clim_soil) == len(keys):
                                    for k, v in zip(keys, clim_soil):
                                        if isinstance(v, (int, float)): state['data'][k] = v
                            except Exception as e:
                                print(f"Chat Data Fetch Error: {e}")
                        else:
                            valid = False
                    except:
                        valid = False
                
                # Check for "Auto" in both original and translated forms
                is_auto_fail = False
                if not valid:
                    if 'auto' in msg.lower():
                        is_auto_fail = True
                    elif lang_code != 'English':
                         # Check if user msg matches translated "Auto"
                         try:
                             trans_auto = translator_fn('Auto', lang_code)
                             if trans_auto and trans_auto.lower() in msg.lower():
                                 is_auto_fail = True
                         except: pass
                
                if is_auto_fail:
                    err_msg = translator_fn("⚠️ Auto-detection failed (Browser blocked location). Please type Latitude and Longitude manually (e.g., 22.5, 88.3).", lang_code)
                    history.append((msg, err_msg))
                    
                    # Ensure choices are translated for re-display (Label, Value)
                    auto_choice_lbl = translator_fn('Auto', lang_code)
                    return history, state, gr.update(visible=True, choices=[(auto_choice_lbl, 'Auto')])

            # B. Float Handling
            elif q_meta['type'] == 'float':
                # Use regex on original message, assuming numbers are universal-ish
                match = re.search(r"[-+]?\d*\.\d+|\d+", msg)
                if match:
                    val = float(match.group())
                    if val >= 0: 
                        state['data'][key] = val
                        valid = True

            # C. Choice Handling
            elif q_meta['type'] == 'choice':
                if key in INTERPRETABLE_MAPS:
                    # 1. Direct English Match
                    for label, val in INTERPRETABLE_MAPS[key].items():
                        if label.lower() == msg.lower():
                            state['data'][key] = val
                            valid = True
                            break
                    
                    # 2. Check against Translated Labels (User sees translated options)
                    if not valid and lang_code != 'English':
                        for label, val in INTERPRETABLE_MAPS[key].items():
                            try:
                                trans_label = translator_fn(label, lang_code)
                                if trans_label.lower() == msg.lower():
                                    state['data'][key] = val
                                    valid = True
                                    break
                            except: continue

                    # 3. Value Match (Fallback)
                    if not valid:
                        if msg in INTERPRETABLE_MAPS[key].values():
                             state['data'][key] = msg
                             valid = True
                
                # 4. Generic Options Check (Original & Translated)
                if not valid and 'options' in q_meta:
                     if msg in q_meta['options']:
                        state['data'][key] = msg
                        valid = True
                     elif lang_code != 'English':
                         # Check translated options
                         for opt in q_meta['options']:
                             try:
                                 trans_opt = translator_fn(opt, lang_code)
                                 if trans_opt.lower() == msg.lower():
                                     state['data'][key] = opt # Store English Key
                                     valid = True
                                     break
                             except: continue

            # D. Date Handling
            elif q_meta['type'] == 'date':
                doy = date_str_to_doy(msg)
                if not np.isnan(doy):
                    state['data'][key] = doy
                    valid = True

            # --- VALIDATION FAILED ---
            if not valid:
                err = translator_fn("⚠️ I didn't understand that. Please try again.", lang_code)
                if not history or history[-1][1] != err:
                    history.append((msg, err))
                
                # Translate options for re-display (Label, Value)
                eng_opts = q_meta.get('options', [])
                trans_opts = [(translator_fn(opt, lang_code), opt) for opt in eng_opts] if eng_opts else []
                
                return history, state, gr.update(visible=True, choices=trans_opts) if trans_opts else gr.update()

            state['step'] += 1

        # --- 2. ASK NEXT QUESTION ---
        if state['step'] < len(CHAT_QUESTIONS):
            next_q = CHAT_QUESTIONS[state['step']]
            q_text = translator_fn(next_q['q'], lang_code)
            
            prev_key = CHAT_QUESTIONS[state['step']-1]['key'] if state['step'] > 0 else None
            
            if prev_key == 'LOC':
                history.append((None, q_text))
            elif step_idx == 0 and is_greeting: # This branch is unlikely now due to logic shift, but kept safe
                history.append((None, q_text))
            else:
                 history.append((msg, q_text))

            # Translate Options for the new question (Label, Value)
            if next_q.get('options'):
                eng_opts = next_q['options']
                trans_opts = [(translator_fn(opt, lang_code), opt) for opt in eng_opts]
                return history, state, gr.update(visible=True, choices=trans_opts, value=None, label="Select")
            else:
                return history, state, gr.update(visible=False, value=None)

        # --- 3. RUN PREDICTION (All steps done) ---
        else:
            wait_msg = translator_fn("🌱 Processing... Running simulation...", lang_code)
            history.append((msg, wait_msg))

            final_inputs = state['data'].copy()

            try:
                area = float(final_inputs.get('CRLPARHA', 1.0))
                if area <= 0: area = 1.0
                total_seed = float(final_inputs.get('SRATE_kg', 30.0))
                final_inputs['SRATEHA'] = total_seed / area
            except:
                final_inputs['SRATEHA'] = 30.0 

            for k, v in ECONOMIC_DEFAULTS.items():
                if k not in final_inputs: final_inputs[k] = v

            # --- NEW: Set Default Optimization Weights for Chat ---
            if 'W_YIELD' not in final_inputs: final_inputs['W_YIELD'] = 1.0
            if 'W_PROFIT' not in final_inputs: final_inputs['W_PROFIT'] = 1.0
            if 'W_ENV' not in final_inputs: final_inputs['W_ENV'] = 1.0
            # ------------------------------------------------------

            try:
                results = predict_fn(final_inputs, {}, lang_code)

                y, n, p, k, rel, advice, plan, table, _ = results

                # Clean strings for better markdown rendering
                table_clean = dedent(str(table)).strip()
                plan_clean = dedent(str(plan)).strip()
                advice_clean = dedent(str(advice)).strip()

                # Ensure strictly explicit newlines for the table to enforce markdown rendering
                report = (
                    f"**✅ Analysis Complete**\n\n"
                    f"**Yield:** {y} | **Reliability:** {rel}\n\n"
                    f"**💊 Recommended Nutrients (Current Practice):**\n"
                    f"- N: {n}\n"
                    f"- P: {p}\n"
                    f"- K: {k}\n\n"
                    f"**📈 Economic Analysis:**\n\n"
                    f"{table_clean}\n\n"
                    f"**🚜 Plan:**\n"
                    f"{plan_clean}\n\n"
                    f"**💡 Advice:**\n"
                    f"{advice_clean}"
                )
                
                history.append((None, report))

                state = {'step': 0, 'data': {}}
                
                start_q = CHAT_QUESTIONS[0]
                start_msg = translator_fn(start_q['q'], lang_code)
                history.append((None, start_msg))
                
                # Translate start options (Label, Value)
                start_opts_eng = start_q.get('options', [])
                start_opts_trans = [(translator_fn(opt, lang_code), opt) for opt in start_opts_eng] if start_opts_eng else []

                return history, state, gr.update(visible=True, choices=start_opts_trans, value=None)

            except Exception as e:
                err_msg = f"Error during simulation: {str(e)}"
                history.append((None, err_msg))
                state = {'step': 0, 'data': {}}
                return history, state, gr.update(visible=True, choices=[('Auto', 'Auto')])

    except Exception as e:
        print(f"Chatbot Error: {e}")
        traceback.print_exc()
        if history is None: history = []
        history.append((None, f"⚠️ System Error: {str(e)}. Please type 'Reset'."))
        return history, state, gr.update()