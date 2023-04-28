# -*- coding: utf-8 -*-
from validators.utils import validator

__all__ = ('es_cif', 'es_nif', 'es_nie', 'es_doi',)


def nif_nie_validation(doi, number_by_letter, special_cases):
    """
    Validate if the doi is a NIF or a NIE.
    :param doi: DOI to validate.
    :return: boolean if it's valid.
    """
    doi = doi.upper()
    if doi in special_cases:
        return False

    table = 'TRWAGMYFPDXBNJZSQVHLCKE'

    if len(doi) != 9:
        return False

    control = doi[8]

    # If it is not a DNI, convert the first letter to the corresponding
    # digit
    numbers = number_by_letter.get(doi[0], doi[0]) + doi[1:8]

    return numbers.isdigit() and control == table[int(numbers) % 23]


@validator
def es_cif(doi):
    """
    Validate a Spanish CIF.

    Each company in Spain prior to 2008 had a distinct CIF and has been
    discontinued. For more information see `wikipedia.org/cif`_.

    The new replacement is to use NIF for absolutely everything. The issue is
    that there are "types" of NIFs now: company, person[citizen vs recident]
    all distinguished by the first character of the DOI. For this reason we
    will continue to call CIF NIFs that are used for companies.

    This validator is based on `generadordni.es`_.

    .. _generadordni.es:
        https://generadordni.es/

    .. _wikipedia.org/cif:
        https://es.wikipedia.org/wiki/C%C3%B3digo_de_identificaci%C3%B3n_fiscal

    Examples::

        >>> es_cif('B25162520')
        True

        >>> es_cif('B25162529')
        ValidationFailure(func=es_cif, args=...)

    .. versionadded:: 0.13.0

    :param doi: DOI to validate
    """
    doi = doi.upper()

    if len(doi) != 9:
        return False

    table = 'JABCDEFGHI'
    first_chr = doi[0]
    doi_body = doi[1:8]
    control = doi[8]

    if not doi_body.isdigit():
        return False

    odd_result = 0
    even_result = 0
    for index, char in enumerate(doi_body):
        if index % 2 == 0:
            # Multiply each each odd position doi digit by 2 and sum it all
            # together
            odd_result += sum(map(int, str(int(char) * 2)))
        else:
            even_result += int(char)

    res = (10 - (even_result + odd_result) % 10) % 10

    if first_chr in 'ABEH':  # Number type
        return str(res) == control
    elif first_chr in 'PSQW':  # Letter type
        return table[res] == control
    elif first_chr not in 'CDFGJNRUV':
        return False

    return control == str(res) or control == table[res]


@validator
def es_nif(doi):
    """
    Validate a Spanish NIF.

    Each entity, be it person or company in Spain has a distinct NIF. Since
    we've designated CIF to be a company NIF, this NIF is only for person.
    For more information see `wikipedia.org/nif`_.

    This validator is based on `generadordni.es`_.

    .. _generadordni.es:
        https://generadordni.es/

    .. _wikipedia.org/nif:
        https://es.wikipedia.org/wiki/N%C3%BAmero_de_identificaci%C3%B3n_fiscal

    Examples::

        >>> es_nif('26643189N')
        True

        >>> es_nif('26643189X')
        ValidationFailure(func=es_nif, args=...)

    .. versionadded:: 0.13.0

    :param doi: DOI to validate
    """
    number_by_letter = {'L': '0', 'M': '0', 'K': '0'}
    special_cases = ['X0000000T', '00000000T', '00000001R']
    return nif_nie_validation(doi, number_by_letter, special_cases)


@validator
def es_nie(doi):
    """
    Validate a Spanish NIE.

    The NIE is a tax identification number in Spain, known in Spanish as the
    NIE, or more formally the Número de identidad de extranjero. For more
    information see `wikipedia.org/nie`_.

    This validator is based on `generadordni.es`_.

    .. _generadordni.es:
        https://generadordni.es/

    .. _wikipedia.org/nie:
        https://es.wikipedia.org/wiki/N%C3%BAmero_de_identidad_de_extranjero

    Examples::

        >>> es_nie('X0095892M')
        True

        >>> es_nie('X0095892X')
        ValidationFailure(func=es_nie, args=...)

    .. versionadded:: 0.13.0

    :param doi: DOI to validate
    """
    number_by_letter = {'X': '0', 'Y': '1', 'Z': '2'}
    special_cases = ['X0000000T']

    # NIE must must start with X Y or Z
    if not doi or doi[0] not in number_by_letter.keys():
        return False

    return nif_nie_validation(doi, number_by_letter, special_cases)


@validator
def es_doi(doi):
    """
    Validate a Spanish DOI.

    A DOI in spain is all NIF / CIF / NIE / DNI -- a digital ID. For more
    information see `wikipedia.org/doi`_.

    This validator is based on `generadordni.es`_.

    .. _generadordni.es:
        https://generadordni.es/

    .. _wikipedia.org/doi:
        https://es.wikipedia.org/wiki/Identificador_de_objeto_digital

    Examples::

        >>> es_doi('X0095892M')
        True

        >>> es_doi('X0095892X')
        ValidationFailure(func=es_doi, args=...)

    .. versionadded:: 0.13.0

    :param doi: DOI to validate
    """
    return es_nie(doi) or es_nif(doi) or es_cif(doi)
