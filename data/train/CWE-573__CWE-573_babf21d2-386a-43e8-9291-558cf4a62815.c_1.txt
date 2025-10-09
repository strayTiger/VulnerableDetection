void CWE253_Incorrect_Check_of_Function_Return_Value__wchar_t_putchar_01_bad()
{
    /* FLAW: putwchar() might fail, in which case the return value will be WEOF (-1), but
     * we are checking to see if the return value is 0 */
    if (putwchar((wchar_t)L'A') == 0)
    {
        printLine("putwchar failed!");
    }
}