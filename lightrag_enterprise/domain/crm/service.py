from __future__ import annotations

from dataclasses import dataclass, replace

from .models import CRMContact, CRMTicket


@dataclass
class InMemoryCRMRepository:
    """Reference repository for tests and early adapters.

    Production deployments should back this contract with a database or CRM
    connector while preserving tenant/workspace keys.
    """

    contacts: dict[str, CRMContact]
    tickets: dict[str, CRMTicket]

    def __init__(self) -> None:
        self.contacts = {}
        self.tickets = {}

    def create_contact(self, contact: CRMContact) -> CRMContact:
        if contact.contact_id in self.contacts:
            raise ValueError("Contact already exists")
        self.contacts[contact.contact_id] = contact
        return contact

    def update_contact(self, contact_id: str, **changes: object) -> CRMContact:
        current = self.contacts[contact_id]
        updated = replace(current, **changes)
        self.contacts[contact_id] = updated
        return updated

    def create_ticket(self, ticket: CRMTicket) -> CRMTicket:
        if ticket.ticket_id in self.tickets:
            raise ValueError("Ticket already exists")
        self.tickets[ticket.ticket_id] = ticket
        return ticket

    def update_ticket(self, ticket_id: str, **changes: object) -> CRMTicket:
        current = self.tickets[ticket_id]
        updated = replace(current, **changes)
        self.tickets[ticket_id] = updated
        return updated
