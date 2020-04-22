import delimited “panel_data_city105_diff_fb13567_bmi.csv”, encoding(GB18030)
xtset code time
gen aver_t3 = (t0+tb1+tb2)/3
gen aver_h3 = (rh0+rhb1+rhb2)/3
gen y3 = log(fb3+1)-log(diff+1)
xtfmb y3 bmi aver_t3 aver_h3 p_gdp p_dns beds old, lag(3) 