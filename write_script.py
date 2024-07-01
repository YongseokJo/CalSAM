import os

def create_bash_script(template_file, output_file, bash_command):

    with open(template_file, 'r') as file:
        template_content = file.read()

    if isinstance(bash_command, list):
        script_content = template_content.replace('$PLACEHOLDER1$',
                                                  bash_command[0])
        script_content = script_content.replace('$PLACEHOLDER2$',
                                                bash_command[1])
        script_content = script_content.replace('$PLACEHOLDER3$',
                                                bash_command[1])
    if isinstance(bash_command, str):
        script_content = template_content.replace('$PLACEHOLDER1$',
                                                  bash_command)

    with open(output_file, 'w') as file:
        file.write(script_content)
        os.chmod(output_file, 0o755)
        print(f"Bash script '{output_file}' created successfully.")



if __name__ == "__main__":
    template_file = 'template_ili.txt'
    template_file = 'template_sam.txt'
    output_file   = 'data/script.sh'
    bash_command  = '-o 10 -p a.txt'
    bash_command  = ['3', '-o 10 -p a.txt']
    create_bash_script(template_file, output_file, bash_command)

