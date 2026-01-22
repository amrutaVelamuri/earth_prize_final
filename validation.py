# validation.py
import streamlit as st

class InputValidator:
    
    @staticmethod
    def validate_coordinate(value, coord_type='latitude'):
        """Validate geographic coordinates"""
        if coord_type == 'latitude':
            if value < -90 or value > 90:
                return False, f"Latitude must be between -90 and 90 (got {value})"
        elif coord_type == 'longitude':
            if value < -180 or value > 180:
                return False, f"Longitude must be between -180 and 180 (got {value})"
        return True, "Valid"
    
    @staticmethod
    def validate_waterfall_params(height, flow):
        """Validate waterfall parameters"""
        if height > 0 and flow == 0:
            return False, "Waterfall height provided but flow rate is 0"
        if flow > 0 and height == 0:
            return False, "Flow rate provided but waterfall height is 0"
        if height < 0 or flow < 0:
            return False, "Waterfall parameters cannot be negative"
        return True, "Valid"
    
    @staticmethod
    def validate_geothermal_params(temp, depth):
        """Validate geothermal parameters"""
        if temp < 0 or temp > 900:
            return False, f"Temperature must be between 0-900°C (got {temp}°C)"
        if depth < 0 or depth > 10:
            return False, f"Depth must be between 0-10 km (got {depth} km)"
        
        # Geological consistency check
        expected_temp = 25 + (depth * 60)  # ~60°C per km
        if temp > expected_temp * 2:
            return False, f"Warning: Temperature {temp}°C seems high for {depth}km depth"
        
        return True, "Valid"