import os
from subprocess import check_call, CalledProcessError

URLBASE = 'https://storage.ramp.studio/mars_craters/{}'
DATA = [
    'images_quad_77.npy', ]
LABELS = [
    'quad77_labels.csv', ]


def download_file(url, output_file=None, shell=False):
    """
    Download a file from a url using `wget`

    Parameters
    ----------
    url : str
        Url to retrive the file
    output_file : str, optional (default is None)
        Local path to write the downloaded file
    shell : bool, optional (default is False)
        `subprocess.check_call` keyword

    """
    command_list = ['wget']

    if output_file is not None:
        command_list.append('-O')
        command_list.append(output_file)

    command_list.append(url)

    print('\nRunning {}\n'.format(' '.join(command_list)))

    check_call(command_list, shell=shell)


def main(output_dir='data'):
    filenames = DATA + LABELS
    urls = [URLBASE.format(filename) for filename in filenames]

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    notfound = []
    for url, filename in zip(urls, filenames):
        output_file = os.path.join(output_dir, filename)

        if os.path.exists(output_file):
            continue

        try:
            download_file(url, output_file)
        except CalledProcessError:
            notfound.append(filename)
            os.remove(output_file)

    if notfound:
        print("The following file could not be downloaded:")
        lis = ["\t{}\n".format(filename) for filename in notfound]
        print("".join(lis))


if __name__ == '__main__':
    main()
