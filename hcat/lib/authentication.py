import json
import os
import os.path

import machineid
import requests

from hcat.gui.key_input import KeyEnterWidget
import hcat


def activate_license(license_key, is_demo=False):
    """ we dont need to activate anymore... """
    # machine_fingerprint = machineid.hashed_id("hcat")
    # os.environ['KEYGEN_ACCOUNT_ID'] = "733ce7bb-483a-4589-adfa-3167723eeaf9"
    # validation = requests.post(
    #     "https://api.keygen.sh/v1/accounts/{}/licenses/actions/validate-key".format(
    #         os.environ['KEYGEN_ACCOUNT_ID']
    #     ),
    #     headers={
    #         "Content-Type": "application/vnd.api+json",
    #         "Accept": "application/vnd.api+json",
    #     },
    #     data=json.dumps(
    #         {
    #             "meta": {
    #                 "scope": {"fingerprint": machine_fingerprint},
    #                 "key": license_key,
    #             }
    #         }
    #     ),
    # ).json()
    #
    # if "errors" in validation:
    #     errs = validation["errors"]
    #
    #
    #     return False, "license validation failed: {}".format(
    #         map(lambda e: "{} - {}".format(e["title"], e["detail"]).lower(), errs)
    #     )
    #
    # if (
    #         validation['data']['attributes']['status'] == 'ACTIVE' and is_demo
    # ):
    #     return True, 'demo lisence is active'
    # # If the license is valid for the current machine, that means it has
    # # already been activated. We can return early.
    # if validation["meta"]["valid"]:  # if its a demo, we can just return true... NOT GREAT!
    #     return True, "license has already been activated on this machine"
    #
    # # Otherwise, we need to determine why the current license is not valid,
    # # because in our case it may be invalid because another machine has
    # # already been activated, or it may be invalid because it doesn't
    # # have any activated machines associated with it yet and in that case
    # # we'll need to activate one.
    # validation_code = validation["meta"]["code"]
    # activation_is_required = (
    #         validation_code == "FINGERPRINT_SCOPE_MISMATCH"
    #         or validation_code == "NO_MACHINES"
    #         or validation_code == "NO_MACHINE"
    # )
    #
    # if not activation_is_required:
    #     return False, "license {}".format(validation["meta"]["detail"])
    #
    # # If we've gotten this far, then our license has not been activated yet,
    # # so we should go ahead and activate the current machine.
    # activation = requests.post(
    #     "https://api.keygen.sh/v1/accounts/{}/machines".format(
    #         os.environ["KEYGEN_ACCOUNT_ID"]
    #     ),
    #     headers={
    #         "Authorization": "License {}".format(license_key),
    #         "Content-Type": "application/vnd.api+json",
    #         "Accept": "application/vnd.api+json",
    #     },
    #     data=json.dumps(
    #         {
    #             "data": {
    #                 "type": "machines",
    #                 "attributes": {"fingerprint": machine_fingerprint},
    #                 "relationships": {
    #                     "license": {
    #                         "data": {"type": "licenses", "id": validation["data"]["id"]}
    #                     }
    #                 },
    #             }
    #         }
    #     ),
    # ).json()
    #
    # # If we get back an error, our activation failed.
    # if "errors" in activation:
    #     errs = activation["errors"]
    #
    #     return False, "license activation failed: {}".format(
    #         ",".join(
    #             map(lambda e: "{} - {}".format(e["title"], e["detail"]).lower(), errs)
    #         )
    #     )

    return True, "license activated"


def authenticate() -> bool:
    filename = os.path.join(hcat.__path__[0], "lisense.key")
    __DEMO_KEY__ = '90A779-E3582F-827C46-7EE725-EDB868-V3'
    # print(f'{filename=}, {os.path.exists(filename)}')
    if not os.path.exists(filename):
        demo_auth, errors = activate_license(__DEMO_KEY__, is_demo=True)
        if not demo_auth:
            # key_enter = KeyEnterWidget()
            key = input("Please enter your lisence key: ")
            auth, errors = activate_license(key)
        else:
            auth = demo_auth
            key = __DEMO_KEY__

        if auth:
            with open(filename, "w") as file:
                file.write(key)
    else:  # assured to exist
        with open(filename, "r") as file:
            key = file.readline()
            auth, errors = activate_license(key)
        if not auth:
            os.remove(filename)

    return auth, errors
