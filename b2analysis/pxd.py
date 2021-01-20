def get_module_data(df, layer, ladder, sensor):
    return df.loc[(df['layer'] == layer) & (df['ladder'] == ladder) & (df['sensor'] == sensor)]
