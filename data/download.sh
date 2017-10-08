#!/usr/bin/env bash
virtualenv congressenv
source congressenv/bin/activate
pip install -r congress/requirements.txt
congress/run fdsys --collections=BILLS --store=mods,xml,text --bulkdata=False
congress/run fdsys --collections=BILLSTATUS --store=mods,xml,text --bulkdata=False
congress/run bills
deactivate
