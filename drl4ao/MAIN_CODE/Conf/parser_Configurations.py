#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json


#%%  SIMULATIONS  CONFIGURATION  FILE


class local_arg:

    def __init__(self):
        pass

user_arg = local_arg()

class ConfigAction(argparse.Action):

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(user_arg, self.dest, values)

class Config:
    # ATTRIBUTS :
    def __init__(self, args):
        # default values
        for arg in vars(args):
            if getattr(args, arg) is not None:
                setattr(self, arg, getattr(args, arg))
        # load config file and overwrite default values
        if args.config is not None:
            with open(os.path.expanduser(args.config), 'r') as f:
                cfg = json.load(f)
            for k, v in cfg.items():
                setattr(self, k, v)
        # overwrite with command line parameters
        for arg in vars(user_arg):
            if getattr(args, arg) is not None:
                setattr(self, arg, getattr(user_arg, arg))
    # MÉTHODE :
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(vars(self), f, indent=2)


#%% END
