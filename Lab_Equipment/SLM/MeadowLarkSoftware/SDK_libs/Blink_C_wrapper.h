//
//:  Blink_SDK_C_wrapper for programming languages that can interface with DLLs
//
//   (c) Copyright Meadowlark Optics 2017, All Rights Reserved.


#ifndef BLINK_C_WRAPPER_H_
#define BLINK_C_WRAPPER_

#ifdef BLINK_C_WRAPPER_EXPORTS
#define BLINK_C_WRAPPER_API __declspec(dllexport)
#else
#define BLINK_C_WRAPPER_API __declspec(dllimport)
#endif


#ifdef __cplusplus
extern "C" { /* using a C++ compiler */
#endif

	// ----------------------------- Create_SDK ----------------------------------
	// Initializes SDK and creates the window to send image data to the SLM.
	//
	// Open a window to write image data to the SLM
	// ---------------------------------------------------------------------------
	BLINK_C_WRAPPER_API void Create_SDK();

	// ----------------------------- Delete_SDK ----------------------------------
	// Destructs the SDK and cleans up
	//
	// ---------------------------------------------------------------------------
	BLINK_C_WRAPPER_API void Delete_SDK();

	// ----------------------------- Write_image ---------------------------------
	// Writes an image to the SLM via the display window. Each pixel needs three
	// bytes (red, green and blue). Only the red channel is used by the SLM but
	// it is recommended to replicate the grayscale data across all three channels
	//
	// returns true if successful, false if there was an error
	// image_data - pointer to byte data of the image (1920 x 1152 x 1- or 3-bytes)
	// is_8_bit - indicates whether image_data is RGB or 8-bit grayscale
	// ---------------------------------------------------------------------------
	BLINK_C_WRAPPER_API int Write_image(unsigned char* image_data, int is_8_bit);

	// ------------------------------ Load_lut -----------------------------------
	// Loads a lookup table to the SLM controller via the USB to serial port.
	//
	// returns true if successful, false if there was an error
	// com_port - the string name of the COM port (i.e. COM3)
	// file_path - the full path to the BLT file
	// ---------------------------------------------------------------------------
	BLINK_C_WRAPPER_API int Load_lut(char* com_port, char* file_path);

	// ----------------------------- Set_channel ---------------------------------
	// Sets HDMI color channel from which the SLM loads image data.
	//
	// returns true if successful, false if there was an error
	// com_port - the string name of the COM port (i.e. COM3)
	// channel - 0 for red, 1 for green or 2 for blue
	// ---------------------------------------------------------------------------
	BLINK_C_WRAPPER_API int Set_channel(const char* com_port, int channel);

	// ----------------------------- Get_SLMTemp ---------------------------------
	// Reads the current SLM temperature
	//
	// com_port - the string name of the COM port (i.e. COM3)
	// returns a floating point number with the current SLM temperature.
	// ---------------------------------------------------------------------------
	BLINK_C_WRAPPER_API float Get_SLMTemp(const char* com_port);

	// ----------------------------- Get_SLMVCom ---------------------------------
	// Reads the coverglass voltage
	//
	// com_port - the string name of the COM port (i.e. COM3)
	// returns a floating point number with the coverglass voltage
	// ---------------------------------------------------------------------------
	BLINK_C_WRAPPER_API float Get_SLMVCom(const char* com_port);

	// ----------------------------- Set_SLMVCom ---------------------------------
	// Sets the coverglass voltage
	//
	// com_port - the string name of the COM port (i.e. COM3)
	// volts - the floating point voltage to set coverglass to
	// returns true if successful, false if there was an error
	// ---------------------------------------------------------------------------
	BLINK_C_WRAPPER_API int Set_SLMVCom(const char* com_port, float volts);

#ifdef __cplusplus
}
#endif

#endif //BLINK_C_WRAPPER_